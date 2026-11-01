"""
Voice assistant for Gesture-Controlled Virtual Mouse.

Architecture
------------
Microphone  →  sounddevice RawInputStream  →  audio_queue
audio_queue →  _audio_chunks() generator   →  client.stream()  (SDK write thread)
AssemblyAI  →  SDK read thread  →  on_turn callback
on_turn     →  turn_order dedup  →  insertion_queue
insertion_queue  →  _insertion_worker thread  →  _type_text()  →  Notepad / target app

Key fixes (v3)
--------------
1. **turn_order deduplication** – TurnEvent.turn_order is a stable integer that
   uniquely identifies each speaker turn.  The same turn_order can arrive many
   times (partials + refined finals).  We keep a set of already-processed
   turn_orders and skip every re-delivery once a turn_order has been committed.
2. **Serialised insertion worker** – a single dedicated thread drains the
   insertion_queue so no two clipboard pastes ever overlap.
3. **Stable clipboard helper** – one persistent tkinter.Tk root is created once
   at startup and reused for all clipboard operations, avoiding the expensive
   create/destroy cycle that caused race conditions and latency spikes.
4. **Dictation text is never passed through _normalize's punctuation stripper**
   during insertion – only the command-matching path normalises.  Raw transcript
   text (with original punctuation) goes to the target application.
5. **No Backspace after paste** – the trailing space separating consecutive turns
   is intentional and is never removed.
6. **Duplicate-guard uses turn_order** as primary key; a text-based guard is
   kept as a secondary defence for edge cases where turn_order alone is not
   enough (e.g. a reconnect that replays the last turn with a new turn_order).
"""

import os
import queue
import re
import subprocess
import sys
import threading
import time
import webbrowser
from pathlib import Path

import pyautogui
import pyttsx3
import sounddevice as sd

from assemblyai.streaming.v3 import (
    RealTimeEvents,
    RealTimeParameters,
    RealTimeTranscriber,
    RealTimeTranscriberOptions,
)

from voice_stats import (
    record_recognition,
    record_command,
    record_ignored,
    record_dictation,
    record_error,
)

ROOT = Path(__file__).resolve().parent.parent
KEYBOARD_PATH = ROOT / "src" / "virtual_keyboard.py"

RATE = 16000
BLOCK = 4000

try:
    INPUT_DEVICE = int(os.environ.get("VOICE_INPUT_DEVICE", "1"))
except ValueError:
    INPUT_DEVICE = 1

# Key terms improve recognition of project-specific vocabulary and app commands.
KEYTERMS = [
    "Notepad",
    "note pad",
    "Calculator",
    "close Notepad",
    "close Calculator",
    "close Chrome",
    "close this window",
    "calculate",
    "plus",
    "minus",
    "times",
    "divided by",
    "Google Chrome",
    "Chrome",
    "browser",
    "voice help",
    "start dictation",
    "stop dictation",
    "left click",
    "right click",
    "double click",
    "scroll up",
    "scroll down",
    "select all",
    "backspace",
    "virtual keyboard",
    "gesture controller",
    "AssemblyAI",
]


class VoiceAssistant:
    """Cloud speech recognition with local command execution and dictation."""

    def __init__(self, speak=True):
        self.speak_enabled = speak
        self.running = False
        self.dictation_mode = False
        self.audio_queue: queue.Queue[bytes] = queue.Queue(maxsize=30)
        self.keyboard_process = None
        self.stream = None
        self.worker = None
        self.transcriber = None
        self.tts = None
        self.tts_lock = threading.Lock()

        # ---- Duplicate / ordering state --------------------------------
        # Set of turn_orders that have already been processed (primary guard).
        self._processed_turns: set[int] = set()
        self._turns_lock = threading.Lock()

        # Secondary text-based guard: (normalized_text, monotonic_timestamp).
        # Used for reconnect replays that arrive under a fresh turn_order.
        self._last_committed_text = ""
        self._last_committed_time = 0.0
        # Minimum seconds before the exact same text is re-accepted from a
        # *different* turn_order (prevents replay duplicates, but is long enough
        # that legitimate repeated sentences still go through).
        self._min_repeat_gap = 4.0

        # ---- Insertion worker ------------------------------------------
        # All text to be typed into the target application goes through this
        # queue so that clipboard operations are never concurrent.
        self._insertion_queue: queue.Queue[str] = queue.Queue()
        self._insertion_worker_thread: threading.Thread | None = None

        # ---- Tkinter clipboard root -----------------------------------
        # Created once; reused for every paste to avoid per-call overhead.
        self._tk_root = None
        self._tk_lock = threading.Lock()

        # ---- TTS echo suppression ------------------------------------
        self._ignore_recognition_until = 0.0

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self):
        if self.running:
            return
        if not os.environ.get("ASSEMBLYAI_API_KEY", "").strip():
            msg = (
                "ASSEMBLYAI_API_KEY is missing. Set it in the PowerShell "
                "session used to launch the gesture controller."
            )
            record_error(msg)
            raise RuntimeError(msg)

        if self.speak_enabled:
            try:
                self.tts = pyttsx3.init()
                self.tts.setProperty("rate", 165)
            except Exception as exc:
                record_error("TTS init: " + str(exc))
                self.tts = None

        # Flush stale audio.
        while not self.audio_queue.empty():
            try:
                self.audio_queue.get_nowait()
            except queue.Empty:
                break

        # Reset deduplication state for a clean start.
        with self._turns_lock:
            self._processed_turns.clear()
        self._last_committed_text = ""
        self._last_committed_time = 0.0

        self._init_tk_root()

        self.running = True

        # Start serialised insertion worker.
        self._insertion_worker_thread = threading.Thread(
            target=self._insertion_worker, daemon=True
        )
        self._insertion_worker_thread.start()

        # Start main audio / transcription thread.
        self.worker = threading.Thread(target=self._listen, daemon=True)
        self.worker.start()

        print("[VOICE] AssemblyAI real-time assistant starting.")
        print("[VOICE] Input device index:", INPUT_DEVICE)
        print("[VOICE] Speech audio is sent to AssemblyAI for transcription.")

    def stop(self):
        self.running = False
        self.dictation_mode = False
        try:
            if self.stream:
                self.stream.stop()
                self.stream.close()
        except Exception as exc:
            print("[VOICE] Microphone shutdown:", exc)
        self.stream = None
        if self.keyboard_process and self.keyboard_process.poll() is None:
            try:
                self.keyboard_process.terminate()
            except Exception:
                pass
        self.keyboard_process = None
        self._destroy_tk_root()

    # ------------------------------------------------------------------
    # Tkinter clipboard root (long-lived, reused)
    # ------------------------------------------------------------------

    def _init_tk_root(self):
        """Create a hidden Tk root for clipboard access.  Called once at start."""
        try:
            import tkinter as tk
            with self._tk_lock:
                if self._tk_root is None:
                    self._tk_root = tk.Tk()
                    self._tk_root.withdraw()
        except Exception as exc:
            print("[VOICE] Could not create Tk clipboard root:", exc)
            self._tk_root = None

    def _destroy_tk_root(self):
        try:
            with self._tk_lock:
                if self._tk_root is not None:
                    self._tk_root.destroy()
                    self._tk_root = None
        except Exception:
            pass

    # ------------------------------------------------------------------
    # TTS
    # ------------------------------------------------------------------

    def say(self, phrase):
        print("[VOICE ASSISTANT]", phrase)
        if self.tts is None:
            return
        # Suppress microphone echo while the assistant speaks.
        estimated_speech_seconds = max(2.0, len(str(phrase)) / 12.0 + 1.0)
        self._ignore_recognition_until = max(
            self._ignore_recognition_until,
            time.monotonic() + estimated_speech_seconds,
        )

        def speak_worker():
            with self.tts_lock:
                try:
                    self.tts.say(phrase)
                    self.tts.runAndWait()
                except Exception as exc:
                    record_error("TTS: " + str(exc))

        threading.Thread(target=speak_worker, daemon=True).start()

    # ------------------------------------------------------------------
    # Audio streaming
    # ------------------------------------------------------------------

    def _audio_callback(self, indata, frames, time_info, status):
        if status:
            print("[VOICE AUDIO]", status)
        if not self.running:
            return
        chunk = bytes(indata)
        try:
            self.audio_queue.put_nowait(chunk)
        except queue.Full:
            try:
                self.audio_queue.get_nowait()
                self.audio_queue.put_nowait(chunk)
            except (queue.Empty, queue.Full):
                pass

    def _audio_chunks(self):
        """Yield PCM blocks to the AssemblyAI WebSocket write thread."""
        while self.running:
            try:
                yield self.audio_queue.get(timeout=0.25)
            except queue.Empty:
                continue

    # ------------------------------------------------------------------
    # Main listener thread
    # ------------------------------------------------------------------

    
    
    def _listen(self):
        api_key = os.environ.get("ASSEMBLYAI_API_KEY", "").strip()
        client = None
        connected = False

        if not api_key:
            print("[VOICE ERROR] AssemblyAI API key is missing.")
            self.running = False
            return

        try:
            client = RealTimeTranscriber(
                RealTimeTranscriberOptions(
                    api_key=api_key,
                    connect_timeout=15.0,
                    max_connection_retries=3,
                    connection_retry_delay=2.0,
                )
            )

            self.transcriber = client

            def on_turn(transcriber, event):
                self._handle_turn_event(event)

            def on_error(transcriber, error):
                message = str(error)
                record_error("AssemblyAI streaming error: " + message)
                print("[VOICE ERROR] AssemblyAI:", message)

            client.on(RealTimeEvents.Turn, on_turn)
            client.on(RealTimeEvents.Error, on_error)

            print("[VOICE] Connecting to AssemblyAI...")

        
            client.connect(
                RealTimeParameters(
                    sample_rate=RATE,
                    speech_model="universal-3-6-pro",
                    language_codes=["en"],
                    mode="min_latency",
                    keyterms_prompt=KEYTERMS,
                )
            )


            # The SDK starts these threads only after a successful handshake.
            if not client._read_thread.is_alive():
                print("[VOICE ERROR] WebSocket connection was not established.")
                return

            connected = True
            print("[VOICE] AssemblyAI connection established.")
            print("[VOICE] Microphone device:", INPUT_DEVICE)

            with sd.RawInputStream(
                samplerate=RATE,
                blocksize=BLOCK,
                dtype="int16",
                channels=1,
                device=INPUT_DEVICE,
                callback=self._audio_callback,
            ) as stream:
                self.stream = stream
                print("[VOICE] Microphone stream opened. Say 'voice help'.")
                client.stream(self._audio_chunks())

        except Exception as exc:
            record_error("Mic/AssemblyAI: " + str(exc))
            print("[VOICE ERROR] Microphone/AssemblyAI:", exc)

            if "401" in str(exc) or "Unauthorized" in str(exc):
                print("[VOICE HELP] Check your AssemblyAI API key.")
            elif "keyterms_prompt" in str(exc):
                print("[VOICE HELP] Check key-term support for your SDK/model.")

        finally:
            self.running = False

            try:
                if self.stream:
                    self.stream.stop()
                    self.stream.close()
            except Exception:
                pass

            self.stream = None

            if client is not None:
                try:
                    client.disconnect(terminate=False)
                except Exception:
                    pass

            self.transcriber = None

            if connected:
                print("[VOICE] AssemblyAI connection closed.")

            print("[VOICE] Assistant stopped.")

    def _listen(self):
        api_key = os.environ.get("ASSEMBLYAI_API_KEY", "").strip()
        client = None
        connected = False

        if not api_key:
            print("[VOICE ERROR] AssemblyAI API key is missing.")
            self.running = False
            return

        try:
            client = RealTimeTranscriber(
                RealTimeTranscriberOptions(
                    api_key=api_key,
                    connect_timeout=15.0,
                    max_connection_retries=3,
                    connection_retry_delay=2.0,
                )
            )

            self.transcriber = client

            def on_turn(transcriber, event):
                self._handle_turn_event(event)

            def on_error(transcriber, error):
                message = str(error)
                record_error("AssemblyAI streaming error: " + message)
                print("[VOICE ERROR] AssemblyAI:", message)

            client.on(RealTimeEvents.Turn, on_turn)
            client.on(RealTimeEvents.Error, on_error)

            print("[VOICE] Connecting to AssemblyAI...")

            client.connect(
                RealTimeParameters(
                    sample_rate=RATE,
                    speech_model="universal-3-6-pro",
                    keyterms_prompt=KEYTERMS,
                )
            )

            # The SDK starts these threads only after a successful handshake.
            if not client._read_thread.is_alive():
                print("[VOICE ERROR] WebSocket connection was not established.")
                return

            connected = True
            print("[VOICE] AssemblyAI connection established.")
            print("[VOICE] Microphone device:", INPUT_DEVICE)

            with sd.RawInputStream(
                samplerate=RATE,
                blocksize=BLOCK,
                dtype="int16",
                channels=1,
                device=INPUT_DEVICE,
                callback=self._audio_callback,
            ) as stream:
                self.stream = stream
                print("[VOICE] Microphone stream opened. Say 'voice help'.")
                client.stream(self._audio_chunks())

        except Exception as exc:
            record_error("Mic/AssemblyAI: " + str(exc))
            print("[VOICE ERROR] Microphone/AssemblyAI:", exc)

            if "401" in str(exc) or "Unauthorized" in str(exc):
                print("[VOICE HELP] Check your AssemblyAI API key.")
            elif "keyterms_prompt" in str(exc):
                print("[VOICE HELP] Check key-term support for your SDK/model.")

        finally:
            self.running = False

            try:
                if self.stream:
                    self.stream.stop()
                    self.stream.close()
            except Exception:
                pass

            self.stream = None

            if client is not None:
                try:
                    client.disconnect(terminate=False)
                except Exception:
                    pass

            self.transcriber = None

            if connected:
                print("[VOICE] AssemblyAI connection closed.")

            print("[VOICE] Assistant stopped.")


    # ------------------------------------------------------------------
    # Turn-event handler (called on SDK's _read_thread)
    # ------------------------------------------------------------------

    def _handle_turn_event(self, event):
        """
        Called by the SDK for every TurnEvent — both partials and finals.

        Filtering strategy
        ------------------
        1. Skip every event where end_of_turn is False (partial transcripts).
        2. Read turn_order from the event.  This is an incrementing integer;
           once we have committed a turn_order we must never commit it again,
           even if the server re-sends it with end_of_turn=True.
        3. Apply a secondary text-based guard to catch reconnect replays where
           the same text arrives under a *new* turn_order within a few seconds.
        4. Pass the raw transcript (with its original capitalisation and
           punctuation) to handle_text / _type_text without stripping.
        """
        # --- 1. Require end_of_turn ----------------------------------------
        if not getattr(event, "end_of_turn", False):
            return

        raw_text = (getattr(event, "transcript", "") or "").strip()
        if not raw_text or not self.running:
            return

        turn_order = getattr(event, "turn_order", None)

        # --- 2. turn_order deduplication ------------------------------------
        if turn_order is not None:
            with self._turns_lock:
                if turn_order in self._processed_turns:
                    print(
                        f"[VOICE DEBUG] Duplicate turn_order={turn_order} skipped: "
                        f"{raw_text!r}"
                    )
                    return
                self._processed_turns.add(turn_order)

        # --- 3. Secondary text-based guard (reconnect replay protection) ------
        normalized = self._normalize(raw_text)
        now = time.monotonic()
        if (
            normalized == self._last_committed_text
            and now - self._last_committed_time < self._min_repeat_gap
        ):
            print(
                f"[VOICE DEBUG] Text replay skipped (gap={now - self._last_committed_time:.2f}s): "
                f"{raw_text!r}"
            )
            return

        self._last_committed_text = normalized
        self._last_committed_time = now

        record_recognition(normalized, 0.0)
        print(
            f"[VOICE] turn_order={turn_order}  end_of_turn=True  text={raw_text!r}"
        )

        # --- 4. Suppress echo of assistant's own speech --------------------
        if time.monotonic() < self._ignore_recognition_until:
            print("[VOICE DEBUG] Ignored (echo suppression):", repr(normalized))
            return

        # --- 5. Route to command handler or dictation ----------------------
        try:
            self.handle_text(raw_text)
        except Exception as exc:
            record_error("Command handler failed: " + str(exc))
            print("[VOICE ERROR] Command handler:", exc)

    # ------------------------------------------------------------------
    # Text helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _normalize(text):
        """Lower-case, strip surrounding punctuation and collapse whitespace.

        Only used for *command matching*, NOT for the text that is typed into
        the target application (which keeps original capitalisation / punctuation).
        """
        return " ".join(text.lower().strip().strip(" .,!?:;").split())

    @staticmethod
    def _contains_any(text, phrases):
        return any(phrase in text for phrase in phrases)

    # ------------------------------------------------------------------
    # Window-focus helpers
    # ------------------------------------------------------------------

    def _focus_notepad(self, timeout=3.0):
        """Best-effort foreground focus for a visible Windows Notepad window."""
        if os.name != "nt":
            return False
        try:
            import ctypes
            from ctypes import wintypes
            user32 = ctypes.windll.user32
            found = {"hwnd": None}
            enum_proc_type = ctypes.WINFUNCTYPE(
                wintypes.BOOL, wintypes.HWND, wintypes.LPARAM
            )

            def callback(hwnd, _):
                if not user32.IsWindowVisible(hwnd):
                    return True
                length = user32.GetWindowTextLengthW(hwnd)
                if length <= 0:
                    return True
                title = ctypes.create_unicode_buffer(length + 1)
                user32.GetWindowTextW(hwnd, title, length + 1)
                if "notepad" in title.value.lower():
                    found["hwnd"] = hwnd
                    return False
                return True

            deadline = time.time() + timeout
            while time.time() < deadline:
                user32.EnumWindows(enum_proc_type(callback), 0)
                if found["hwnd"]:
                    user32.ShowWindow(found["hwnd"], 5)
                    user32.SetForegroundWindow(found["hwnd"])
                    time.sleep(0.2)
                    return True
                time.sleep(0.1)
        except Exception as exc:
            print("[VOICE] Notepad focus attempt failed:", exc)
        return False

    def _open_notepad(self):
        try:
            subprocess.Popen(["notepad.exe"], shell=False)
            record_command("open notepad")
            print("[VOICE] Notepad launch requested.")
            self._focus_notepad()
            self.say("Notepad opened. You can start dictation.")
        except Exception as exc:
            record_error("Notepad launch failed: " + str(exc))
            print("[VOICE ERROR] Notepad launch failed:", exc)
            self.say("Could not open Notepad.")

    def _focus_calculator(self, timeout=3.0):
        """Find and foreground an existing Windows Calculator window."""
        if os.name != "nt":
            return False
        try:
            import ctypes
            from ctypes import wintypes
            user32 = ctypes.windll.user32
            found = {"hwnd": None}
            callback_type = ctypes.WINFUNCTYPE(
                wintypes.BOOL, wintypes.HWND, wintypes.LPARAM
            )

            def callback(hwnd, _):
                if not user32.IsWindowVisible(hwnd):
                    return True
                length = user32.GetWindowTextLengthW(hwnd)
                if length <= 0:
                    return True
                title = ctypes.create_unicode_buffer(length + 1)
                user32.GetWindowTextW(hwnd, title, length + 1)
                if "calculator" in title.value.lower():
                    found["hwnd"] = hwnd
                    return False
                return True

            callback_ref = callback_type(callback)
            deadline = time.time() + timeout
            while time.time() < deadline:
                user32.EnumWindows(callback_ref, 0)
                if found["hwnd"]:
                    user32.ShowWindow(found["hwnd"], 5)
                    user32.SetForegroundWindow(found["hwnd"])
                    time.sleep(0.25)
                    return True
                time.sleep(0.1)
        except Exception as exc:
            print("[VOICE] Calculator focus attempt failed:", exc)
        return False

    def _ensure_calculator(self):
        """Reuse Calculator if open; otherwise launch one instance."""
        if self._focus_calculator(timeout=0.2):
            return True
        try:
            subprocess.Popen(["calc.exe"], shell=False)
            if self._focus_calculator(timeout=3.0):
                return True
            time.sleep(0.4)
            return True
        except Exception as exc:
            record_error("Calculator launch failed: " + str(exc))
            print("[VOICE ERROR] Calculator launch failed:", exc)
            return False

    def _open_calculator(self):
        try:
            if self._ensure_calculator():
                record_command("open calculator")
                self.say("Calculator is ready.")
            else:
                self.say("Could not open Calculator.")
        except Exception as exc:
            record_error("Calculator launch failed: " + str(exc))
            self.say("Could not open Calculator.")

    def _open_browser(self):
        try:
            webbrowser.open("https://www.google.com")
            record_command("open browser")
            self.say("Opening browser.")
        except Exception as exc:
            record_error("Browser launch failed: " + str(exc))
            self.say("Could not open the browser.")
            
    # ------------------------------------------------------------------
    # Open applications, files and folders
    # ------------------------------------------------------------------

    def _handle_open_resource_command(self, text):
        """Open approved Windows applications, files and folders."""
        if os.name != "nt":
            return False

        normalized = self._normalize(text)

        # Keep existing application handlers responsible for these.
        existing_apps = {
            "open notepad", "open the notepad", "launch notepad",
            "start notepad", "open note pad", "launch note pad",
            "open calculator", "launch calculator", "start calculator",
            "open browser", "open chrome", "open google chrome",
            "launch chrome", "start browser", "launch browser",
            "open google", "open edge", "open microsoft edge",
        }

        if normalized in existing_apps:
            return False

        # Approved applications and their Windows launch commands.
        applications = {
            "paint": "mspaint.exe",
            "microsoft paint": "mspaint.exe",
            "word": r"C:\PROGRA~2\MICROS~2\Office12\WINWORD.EXE",
            "microsoft word": r"C:\PROGRA~2\MICROS~2\Office12\WINWORD.EXE",
            "excel": "excel.exe",
            "microsoft excel": "excel.exe",
            "powerpoint": "powerpnt.exe",
            "microsoft powerpoint": "powerpnt.exe",
            "task manager": "taskmgr.exe",
            "command prompt": "cmd.exe",
            "file explorer": "explorer.exe",
            "explorer": "explorer.exe",
            "settings": "ms-settings:",
            "vscode": "code",
            "visual studio code": "code",
        }

        app_match = re.fullmatch(
            r"(?:please )?(?:open|launch|start) (?:the )?(.+)",
            normalized
        )

        if app_match:
            app_name = app_match.group(1).strip()

            if app_name in applications:
                try:
                    launch_target = applications[app_name]

                    if launch_target.endswith(":"):
                        os.startfile(launch_target)
                    elif launch_target == "code":
                        subprocess.Popen(["code"])
                    else:
                        subprocess.Popen([launch_target])

                    record_command("open " + app_name)
                    self.say("Opening " + app_name + ".")
                except Exception as exc:
                    record_error("Application launch failed: " + str(exc))
                    print("[VOICE ERROR] Application launch failed:", exc)
                    self.say("Could not open " + app_name + ".")

                return True

        # Common Windows user folders.
        home = Path.home()

        folders = {
            "desktop": home / "Desktop",
            "my desktop": home / "Desktop",
            "documents": home / "Documents",
            "my documents": home / "Documents",
            "downloads": home / "Downloads",
            "my downloads": home / "Downloads",
            "pictures": home / "Pictures",
            "photos": home / "Pictures",
            "videos": home / "Videos",
            "music": home / "Music",
        }

        folder_match = re.fullmatch(
            r"(?:please )?(?:open|show|go to) (?:the )?(.+)",
            normalized
        )

        if folder_match:
            folder_name = folder_match.group(1).strip()

            if folder_name in folders:
                folder_path = folders[folder_name]

                if folder_path.is_dir():
                    try:
                        os.startfile(str(folder_path))
                        record_command("open folder " + folder_name)
                        self.say("Opening " + folder_name + ".")
                    except Exception as exc:
                        record_error("Folder opening failed: " + str(exc))
                        self.say("Could not open that folder.")
                else:
                    self.say("That folder does not exist.")

                return True

        # Open a file by full path or filename.
        file_match = re.fullmatch(
            r"(?:please )?open (?:the )?file (.+)",
            text.strip(),
            flags=re.IGNORECASE
        )

        if file_match:
            requested = file_match.group(1).strip().strip('"').strip("'")

            try:
                file_path = Path(requested).expanduser()

                # Check the supplied full or relative path first.
                if not file_path.is_absolute():
                    file_path = Path.cwd() / file_path

                if file_path.is_file():
                    os.startfile(str(file_path))
                    record_command("open file " + file_path.name)
                    self.say("Opening " + file_path.name + ".")
                    return True

                # Search the top level of common user folders.
                search_folders = [
                    home / "Desktop",
                    home / "Documents",
                    home / "Downloads",
                    home / "Pictures",
                    home / "Videos",
                ]

                filename = Path(requested).name
                found_file = None

                for folder in search_folders:
                    candidate = folder / filename
                    if candidate.is_file():
                        found_file = candidate
                        break

                if found_file:
                    os.startfile(str(found_file))
                    record_command("open file " + filename)
                    self.say("Opening " + filename + ".")
                else:
                    self.say(
                        "I couldn't find that file. "
                        "Please provide its full path."
                    )
                    record_ignored()

            except Exception as exc:
                record_error("File opening failed: " + str(exc))
                print("[VOICE ERROR] File opening failed:", exc)
                self.say("I couldn't open that file.")

            return True

        return False


    # ------------------------------------------------------------------
    # Command matchers
    # ------------------------------------------------------------------

    def _is_notepad_command(self, text):
        text = self._normalize(text)
        return text in {
            "open notepad", "open the notepad", "launch notepad",
            "start notepad", "open note pad", "launch note pad",
        }

    def _is_browser_command(self, text):
        text = self._normalize(text)
        return text in {
            "open browser", "open chrome", "open google chrome",
            "launch chrome", "start browser", "launch browser",
            "open google", "open edge", "open microsoft edge",
        }

    def _is_calculator_command(self, text):
        text = self._normalize(text)
        return text in {
            "open calculator", "launch calculator", "start calculator",
        }

    def _handle_app_command(self, text):
        """Handle app launches; works even during dictation."""
        if self._is_notepad_command(text):
            self._open_notepad()
            return True
        if self._is_calculator_command(text):
            self._open_calculator()
            return True
        if self._is_browser_command(text):
            self._open_browser()
            return True
        return False

    # ------------------------------------------------------------------
    # Calculation
    # ------------------------------------------------------------------

    @staticmethod
    def _spoken_number_to_digits(phrase):
        """Convert a simple spoken English number phrase to digits."""
        words = phrase.lower().strip().replace("-", " ").split()
        if not words:
            return phrase
        small = {
            "zero": 0, "oh": 0, "one": 1, "two": 2, "three": 3,
            "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8,
            "nine": 9, "ten": 10, "eleven": 11, "twelve": 12,
            "thirteen": 13, "fourteen": 14, "fifteen": 15,
            "sixteen": 16, "seventeen": 17, "eighteen": 18,
            "nineteen": 19, "twenty": 20, "thirty": 30, "forty": 40,
            "fifty": 50, "sixty": 60, "seventy": 70, "eighty": 80,
            "ninety": 90,
        }
        allowed = set(small) | {"and", "hundred", "thousand", "point"}
        if not all(word in allowed for word in words):
            return phrase
        total = 0
        current = 0
        decimal = False
        decimal_digits: list[str] = []
        for word in words:
            if word == "and":
                continue
            if word == "point":
                decimal = True
                continue
            value = small.get(word)
            if value is not None:
                if decimal:
                    decimal_digits.extend(list(str(value)))
                else:
                    current += value
            elif word == "hundred":
                current = max(current, 1) * 100
            elif word == "thousand":
                total += max(current, 1) * 1000
                current = 0
        number = total + current
        if decimal:
            return str(number) + "." + ("".join(decimal_digits) or "0")
        return str(number)

    def _handle_calculation(self, text):
        """Send an explicitly spoken arithmetic expression to Windows Calculator."""
        normalized = self._normalize(text)
                # Remove repeated calculation phrases from a single transcript.
        parts = re.split(
            r"[.!?]\s+(?=(?:calculate|compute|what is|how much is)\b)",
            normalized,
            maxsplit=1,
        )
        normalized = parts[0].strip()
        prefixes = ("calculate ", "compute ", "what is ", "how much is ")
        prefix = next((p for p in prefixes if normalized.startswith(p)), None)
        if prefix is None:
            return False
        expression = normalized[len(prefix):].strip().rstrip(" ?.")
        replacements = (
            (r"\bmultiplied\s+by\b", "*"),
            (r"\bdivided\s+by\b", "/"),
            (r"\bdivide\s+by\b", "/"),
            (r"\btimes\b", "*"),
            (r"\bplus\b", "+"),
            (r"\bminus\b", "-"),
            (r"\bover\b", "/"),
            (r"\bmodulo\b", "%"),
            (r"\bpercent\b", "%"),
        )
        for pattern, replacement in replacements:
            expression = re.sub(pattern, replacement, expression)
        pieces = re.split(r"([+\-*/()%])", expression)
        for index, piece in enumerate(pieces):
            stripped = piece.strip()
            if stripped and not re.fullmatch(r"[0-9.]+", stripped):
                pieces[index] = self._spoken_number_to_digits(stripped)
        expression = "".join(pieces).strip()
        if not expression or not re.fullmatch(r"[0-9\s.+\-*/()%]+", expression):
            self.say("I couldn't understand that calculation. Try saying calculate 25 plus 30.")
            record_ignored()
            return True
        try:
            if not self._ensure_calculator():
                self.say("Could not open Calculator.")
                return True
            pyautogui.press("esc")
            pyautogui.write(expression, interval=0.04)
            pyautogui.press("enter")
            record_command("calculate " + expression)
            self.say("Calculation entered into Calculator.")
        except Exception as exc:
            record_error("Calculator operation failed: " + str(exc))
            print("[VOICE ERROR] Calculator operation failed:", exc)
            self.say("I couldn't enter that calculation.")
        return True

    # ------------------------------------------------------------------
    # Close commands
    # ------------------------------------------------------------------

   
    def _handle_close_command(self, text):
        """Close a named application using normal Windows window controls."""
        normalized = self._normalize(text)

        if normalized in (
            "close this window",
            "close current window",
            "close active window"
        ):
            pyautogui.hotkey("alt", "f4")
            record_command("close current window")
            self.say("Closing the active window.")
            return True

        match = re.fullmatch(
            r"(?:please )?(?:close|quit|exit) (?:the )?(.+)",
            normalized
        )

        if not match:
            return False

        target = match.group(1).strip()

        if target in ("tab", "this tab", "current tab"):
            pyautogui.hotkey("ctrl", "w")
            record_command("close tab")
            self.say("Closing tab.")
            return True

        aliases = {
            "the notepad": "notepad",
            "note pad": "notepad",
            "the calculator": "calculator",
            "the browser": "browser",
            "google": "chrome",
            "google chrome": "chrome",
            "photos": "pictures",
            "my documents": "documents",
            "my downloads": "downloads",
            "vscode": "visual studio code",
            "explorer": "file explorer",
        }

        target = aliases.get(target, target)

        targets = {
            "notepad": ("notepad",),
            "calculator": ("calculator",),
            "chrome": ("chrome", "google chrome"),
            "browser": ("chrome", "google chrome", "microsoft edge", "firefox"),
            "edge": ("microsoft edge",),
            "microsoft edge": ("microsoft edge",),
            "firefox": ("firefox",),
            "paint": ("paint",),
            "word": ("word",),
            "excel": ("excel",),
            "powerpoint": ("powerpoint",),
            "task manager": ("task manager",),
            "command prompt": ("command prompt",),
            "file explorer": ("file explorer",),
            "visual studio code": ("visual studio code",),
            "desktop": ("desktop",),
            "documents": ("documents",),
            "downloads": ("downloads",),
            "pictures": ("pictures",),
            "videos": ("videos",),
            "music": ("music",),
        }

        if target == "virtual keyboard":
            if self.keyboard_process and self.keyboard_process.poll() is None:
                self.keyboard_process.terminate()
                self.keyboard_process = None
                self.say("Virtual keyboard closed.")
            else:
                self.say("Virtual keyboard is not open.")

            record_command("close virtual keyboard")
            return True

        if target not in targets:
            return False

        if os.name != "nt":
            self.say("Closing named applications is supported on Windows.")
            return True

        try:
            import ctypes
            from ctypes import wintypes

            user32 = ctypes.windll.user32
            matching_windows = []

            callback_type = ctypes.WINFUNCTYPE(
                wintypes.BOOL,
                wintypes.HWND,
                wintypes.LPARAM
            )

            wanted = targets[target]

            def callback(hwnd, _):
                if not user32.IsWindowVisible(hwnd):
                    return True

                length = user32.GetWindowTextLengthW(hwnd)

                if length <= 0:
                    return True

                title = ctypes.create_unicode_buffer(length + 1)
                user32.GetWindowTextW(hwnd, title, length + 1)

                lowered = title.value.lower().strip()

                if target == "task manager":
                    if lowered == "task manager":
                        matching_windows.append(hwnd)
                elif any(term in lowered for term in wanted):
                    matching_windows.append(hwnd)

                return True

            callback_ref = callback_type(callback)

            user32.EnumWindows(callback_ref, 0)

            if not matching_windows:
                self.say("I couldn't find an open " + target + " window.")
                record_ignored()
                return True

            
            if target == "task manager":
                hwnd = matching_windows[0]

                print("[VOICE DEBUG] Closing Task Manager HWND:", hwnd)

                # Send a normal Windows close message.
                ctypes.set_last_error(0)
                result = user32.PostMessageW(
                    hwnd,
                    0x0010,  # WM_CLOSE
                    0,
                    0
                )

                print("[VOICE DEBUG] WM_CLOSE result:", result)
                print("[VOICE DEBUG] Windows error code:", ctypes.get_last_error())

                time.sleep(0.5)

                if user32.IsWindow(hwnd):
                    print("[VOICE DEBUG] Task Manager window still exists.")
                else:
                    print("[VOICE DEBUG] Task Manager window closed.")

                # Close it normally, just like pressing Alt + F4.
                pyautogui.hotkey("alt", "f4")

            else:
                # Send the normal Windows close message.
                for hwnd in matching_windows:
                    user32.PostMessageW(hwnd, 0x0010, 0, 0)

            record_command("close " + target)
            self.say("Closing " + target + ".")

        except Exception as exc:
            record_error("Close application failed: " + str(exc))
            print("[VOICE ERROR] Close application failed:", exc)
            self.say("I couldn't close " + target + ".")

        return True

                   
    # ------------------------------------------------------------------
    # Main command router
    # ------------------------------------------------------------------

    def handle_text(self, text):
        """
        Route a finalized transcript to the correct handler.

        ``text`` arrives with its original capitalisation and punctuation.
        Command matching always works on a _normalize()d copy so that
        "Stop dictation." and "stop dictation" both match.
        """
        if not text.strip():
            return

        normalized = self._normalize(text)

         # Commands that are active even while dictation is running.
        if self._handle_close_command(normalized):
            return

        if self._handle_calculation(normalized):
            return

        if self._handle_open_resource_command(text):
            return

        stop_phrases = (
            "stop dictation", "stop the dictation", "end dictation",
            "end the dictation", "finish dictation", "stop typing",
        )

        if self.dictation_mode:
            if self._contains_any(normalized, stop_phrases):
                self.dictation_mode = False
                record_command("stop dictation")
                self.say("Dictation stopped.")
                return

            if self._handle_app_command(normalized):
                return

            if self._contains_any(normalized, ("voice help", "help voice", "list voice commands")):
                self._say_help()
                return

            # Inline dictation control words.
            if normalized in ("new line", "newline", "next line"):
                pyautogui.press("enter")
                record_dictation("\n")
            elif normalized in ("backspace", "delete previous character"):
                pyautogui.press("backspace")
            elif normalized == "space":
                pyautogui.press("space")
                record_dictation(" ")
            elif normalized == "tab":
                pyautogui.press("tab")
            else:
                # Queue the raw text (with punctuation) for serialised insertion.
                self._insertion_queue.put(text)
                record_dictation(text)
            return

        # ---- Out-of-dictation commands ----------------------------------
        commands = {
            "press enter":     lambda: pyautogui.press("enter"),
            "press tab":       lambda: pyautogui.press("tab"),
            "press backspace": lambda: pyautogui.press("backspace"),
            "press escape":    lambda: pyautogui.press("esc"),
            "press space":     lambda: pyautogui.press("space"),
            "left click":      lambda: pyautogui.click(),
            "click":           lambda: pyautogui.click(),
            "right click":     lambda: pyautogui.rightClick(),
            "double click":    lambda: pyautogui.doubleClick(),
            "scroll up":       lambda: pyautogui.scroll(5),
            "scroll down":     lambda: pyautogui.scroll(-5),
            "select all":      lambda: pyautogui.hotkey("ctrl", "a"),
            "copy":            lambda: pyautogui.hotkey("ctrl", "c"),
            "paste":           lambda: pyautogui.hotkey("ctrl", "v"),
            "cut":             lambda: pyautogui.hotkey("ctrl", "x"),
            "undo":            lambda: pyautogui.hotkey("ctrl", "z"),
            "save":            lambda: pyautogui.hotkey("ctrl", "s"),
            "new tab":         lambda: pyautogui.hotkey("ctrl", "t"),
            "close tab":       lambda: pyautogui.hotkey("ctrl", "w"),
        }

        if self._contains_any(normalized, ("voice help", "help voice", "list voice commands")):
            self._say_help()
        elif self._contains_any(normalized, (
            "start dictation", "begin dictation", "dictation on", "start typing",
        )):
            self.dictation_mode = True
            record_command("start dictation")
            self.say("Dictation started. Say stop dictation when finished.")
        elif self._handle_app_command(normalized):
            return
        elif self._contains_any(normalized, (
    "open keyboard",
    "show keyboard",
    "launch keyboard",
    "open virtual keyboard",
    "show virtual keyboard",
    "launch virtual keyboard",
)):
            if self.keyboard_process and self.keyboard_process.poll() is None:
                self.say("Keyboard is already open.")
            elif KEYBOARD_PATH.exists():
                try:
                    self.keyboard_process = subprocess.Popen(
                        [sys.executable, str(KEYBOARD_PATH)], cwd=str(ROOT)
                    )
                    record_command("open keyboard")
                    self.say("Keyboard opened.")
                except Exception as exc:
                    record_error("Keyboard launch failed: " + str(exc))
                    self.say("Could not open the virtual keyboard.")
            else:
                record_error("Missing virtual keyboard: " + str(KEYBOARD_PATH))
                self.say("Virtual keyboard file not found.")
        elif self._contains_any(normalized, (
            "hide keyboard",
            "close keyboard",
            "hide virtual keyboard",
            "close virtual keyboard",
        )):
            if self.keyboard_process and self.keyboard_process.poll() is None:
                self.keyboard_process.terminate()
            self.keyboard_process = None
            record_command("close keyboard")
            self.say("Keyboard closed.")
        elif normalized in commands:
            try:
                commands[normalized]()
                record_command(normalized)
            except Exception as exc:
                record_error("Command failed ({}): {}".format(normalized, exc))
                print("[VOICE ERROR] Command failed:", normalized, exc)
        elif self._contains_any(normalized, ("voice off", "stop voice", "disable voice")):
            record_command("voice off")
            self.say("Voice control stopping.")
            self.stop()
        else:
            record_ignored()
            print("[VOICE] Unmapped phrase ignored:", repr(normalized))

    def _say_help(self):
        record_command("voice help")
        self.say(
            "Say open Notepad, open Calculator, open Chrome, close Notepad, close Calculator, "
            "close Chrome, close this window, calculate 25 plus 30, start dictation, stop dictation, "
            "press enter, select all, copy, paste, undo, save, or voice off."
        )

    # ------------------------------------------------------------------
    # Serialised insertion worker
    # ------------------------------------------------------------------

    def _insertion_worker(self):
        """
        Drain the insertion queue and type each text segment one at a time.

        Running in its own thread guarantees that clipboard operations never
        overlap even if two Turn events fire within milliseconds of each other.
        """
        while self.running or not self._insertion_queue.empty():
            try:
                text = self._insertion_queue.get(timeout=0.3)
            except queue.Empty:
                continue
            self._type_text(text)
            self._insertion_queue.task_done()

    # ------------------------------------------------------------------
    # Text insertion
    # ------------------------------------------------------------------

    
    def _type_text(self, text):
        """Insert recognized speech exactly once using clipboard or fallback."""
        if not text:
            return

        insert_text = text.rstrip() + " "

        try:
            import pyperclip

            pyperclip.copy(insert_text)
            time.sleep(0.15)
            pyautogui.hotkey("ctrl", "v")

            print("[VOICE] Dictated (clipboard):", repr(insert_text))

        except Exception as exc:
            print("[VOICE ERROR] Clipboard paste failed:", exc)

            try:
                pyautogui.write(insert_text, interval=0.02)
                print("[VOICE] Dictated (keyboard fallback):", repr(insert_text))

            except Exception as fallback_exc:
                record_error("Dictation typing failed: " + str(fallback_exc))
                print("[VOICE ERROR] Dictation typing failed:", fallback_exc)

        """
        Paste a finalized transcript into the active application.

        Design decisions
        ----------------
        * The raw transcript text (original capitalisation + punctuation) is
          used verbatim; _normalize() is never applied here.
        * A trailing space is appended so that the next Turn starts a new word
          rather than merging with the previous one.
        * No Backspace is pressed after the paste.
        * The Tkinter root is reused (created once in start()) rather than
          recreated per call, which eliminates the ~40 ms overhead and the
          race conditions that per-call create/destroy caused.
        * If the clipboard path fails the keyboard fallback writes the same
          text; it never calls itself recursively or presses Backspace.
        """
        if not text:
            return

           

