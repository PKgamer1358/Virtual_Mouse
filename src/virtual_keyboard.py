"""
Gesture Virtual Keyboard
========================
Floating Tkinter keyboard that types into any Windows application via
the Win32 SendInput API.

Root-cause fix (v2)
-------------------
The previous version defined _INPUT_UNION with only the 'ki' (KEYBDINPUT)
member.  On 64-bit Windows the correct INPUT struct is 40 bytes, but the
incomplete union produced a struct of only 32 bytes.  SendInput validates
cbSize against its own sizeof(INPUT); a mismatch causes it to return 0 and
send *zero* events.  The fix is to include MOUSEINPUT (32 bytes, the
largest member) in the union so that ctypes sizes the struct correctly.

Key design decisions
--------------------
* No clipboard paste – every character is sent as KEYEVENTF_UNICODE.
  Special keys use virtual-key events.
* AttachThreadInput is used to reliably hand focus to the target window
  before injecting events, then detached immediately after.
* SendInput return value is checked and logged every time.
* The keyboard stays topmost but does NOT steal keyboard focus back from
  the target between keystrokes.
* A Windows named mutex (GestureVKMutex) prevents duplicate instances.
"""

import ctypes
import ctypes.wintypes as wintypes
import sys
import time
import tkinter as tk

# ---------------------------------------------------------------------------
# Optional pywin32 bindings
# ---------------------------------------------------------------------------
try:
    import win32gui
    import win32con
    import win32event
    import win32api
    import win32process
    _WIN32 = True
except ImportError:
    win32gui = win32con = win32event = win32api = win32process = None
    _WIN32 = False

# ---------------------------------------------------------------------------
# Win32 SendInput structures — correctly sized for 64-bit Windows
# ---------------------------------------------------------------------------
# On 64-bit Windows sizeof(INPUT) == 40.
# The union must include MOUSEINPUT (32 bytes) so ctypes computes the right
# size; KEYBDINPUT alone is only 24 bytes which causes a cbSize mismatch
# and makes SendInput silently discard every event.

class MOUSEINPUT(ctypes.Structure):
    """dx, dy, mouseData, dwFlags, time, dwExtraInfo  →  32 bytes (64-bit)."""
    _fields_ = [
        ("dx",          ctypes.c_long),
        ("dy",          ctypes.c_long),
        ("mouseData",   ctypes.c_ulong),
        ("dwFlags",     ctypes.c_ulong),
        ("time",        ctypes.c_ulong),
        ("dwExtraInfo", ctypes.c_void_p),   # ULONG_PTR — pointer-sized
    ]


class KEYBDINPUT(ctypes.Structure):
    """wVk, wScan, dwFlags, time, dwExtraInfo  →  24 bytes (64-bit)."""
    _fields_ = [
        ("wVk",         ctypes.c_ushort),
        ("wScan",       ctypes.c_ushort),
        ("dwFlags",     ctypes.c_ulong),
        ("time",        ctypes.c_ulong),
        ("dwExtraInfo", ctypes.c_void_p),   # ULONG_PTR
    ]


class HARDWAREINPUT(ctypes.Structure):
    _fields_ = [
        ("uMsg",    ctypes.c_ulong),
        ("wParamL", ctypes.c_short),
        ("wParamH", ctypes.c_ushort),
    ]


class _INPUT_UNION(ctypes.Union):
    # MOUSEINPUT (32 bytes) is the largest member → union = 32 bytes.
    _fields_ = [
        ("mi", MOUSEINPUT),
        ("ki", KEYBDINPUT),
        ("hi", HARDWAREINPUT),
    ]


class INPUT(ctypes.Structure):
    # type(4) + pad(4) + union(32) = 40 bytes — matches Windows sizeof(INPUT).
    _fields_ = [
        ("type",  ctypes.c_ulong),
        ("value", _INPUT_UNION),
    ]


_INPUT_SIZE = ctypes.sizeof(INPUT)   # must be 40 on 64-bit Windows

# Sanity-check at import time so the bug is immediately visible.
assert _INPUT_SIZE == 40, (
    f"INPUT struct is {_INPUT_SIZE} bytes; Windows requires 40. "
    "SendInput would silently reject all events."
)

# ---------------------------------------------------------------------------
# SendInput flags and VK codes
# ---------------------------------------------------------------------------
INPUT_KEYBOARD      = 1
KEYEVENTF_EXTENDEDKEY = 0x0001
KEYEVENTF_KEYUP     = 0x0002
KEYEVENTF_UNICODE   = 0x0004
MAPVK_VK_TO_VSC     = 0

VK = {
    "backspace": 0x08,
    "tab":       0x09,
    "enter":     0x0D,
    "caps_lock": 0x14,
    "escape":    0x1B,
    "space":     0x20,
    "delete":    0x2E,
    "left":      0x25,
    "right":     0x27,
    "up":        0x26,
    "down":      0x28,
}

_user32   = ctypes.windll.user32
_kernel32 = ctypes.windll.kernel32


# ---------------------------------------------------------------------------
# Low-level input helpers
# ---------------------------------------------------------------------------

def _ki(vk: int = 0, scan: int = 0, flags: int = 0) -> INPUT:
    """Build one INPUT structure for a keyboard event."""
    inp = INPUT()
    inp.type = INPUT_KEYBOARD
    inp.value.ki.wVk         = vk
    inp.value.ki.wScan       = scan
    inp.value.ki.dwFlags     = flags
    inp.value.ki.time        = 0
    inp.value.ki.dwExtraInfo = 0
    return inp


def _send_inputs(*events: INPUT) -> int:
    """
    Call SendInput and return the number of events actually injected.
    Prints a diagnostic line on failure.
    """
    n   = len(events)
    arr = (INPUT * n)(*events)
    sent = _user32.SendInput(n, arr, _INPUT_SIZE)
    if sent != n:
        err = _kernel32.GetLastError()
        print(f"[VK] SendInput({n}) → {sent} events sent  (LastError={err})")
    else:
        print(f"[VK] SendInput({n}) → OK")
    return sent


def send_unicode_char(char: str) -> int:
    """Send a Unicode character (key-down + key-up) via KEYEVENTF_UNICODE."""
    code = ord(char)
    return _send_inputs(
        _ki(scan=code, flags=KEYEVENTF_UNICODE),
        _ki(scan=code, flags=KEYEVENTF_UNICODE | KEYEVENTF_KEYUP),
    )


def send_vk_pair(vk_code: int) -> int:
    """Send a virtual-key code (key-down + key-up)."""
    scan = _user32.MapVirtualKeyW(vk_code, MAPVK_VK_TO_VSC)
    return _send_inputs(
        _ki(vk=vk_code, scan=scan, flags=0),
        _ki(vk=vk_code, scan=scan, flags=KEYEVENTF_KEYUP),
    )


# ---------------------------------------------------------------------------
# Single-instance mutex
# ---------------------------------------------------------------------------
_MUTEX_NAME   = "GestureVKMutex"
_mutex_handle = None


def _acquire_mutex() -> bool:
    global _mutex_handle
    if not _WIN32:
        return True
    _mutex_handle = win32event.CreateMutex(None, False, _MUTEX_NAME)
    return win32api.GetLastError() != 183   # ERROR_ALREADY_EXISTS == 183


def _release_mutex():
    global _mutex_handle
    if _mutex_handle is not None:
        win32api.CloseHandle(_mutex_handle)
        _mutex_handle = None


# ---------------------------------------------------------------------------
# Virtual Keyboard window
# ---------------------------------------------------------------------------

class VirtualKeyboard:
    """Floating keyboard controlled by the gesture-controlled mouse cursor."""

    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Gesture Virtual Keyboard")
        self.root.geometry("860x420")
        self.root.configure(bg="#111827")
        self.root.attributes("-topmost", True)
        self.root.resizable(False, False)

        self.target_hwnd: int | None = None
        self.target_title: str       = ""
        self.shift_on  = False
        self.caps_on   = False
        self.last_key_time  = 0.0
        self.key_cooldown   = 0.15   # seconds
        self.target_capture_pending = False

        self.preview = tk.StringVar()
        self._build_ui()

        self.root.protocol("WM_DELETE_WINDOW", self.root.withdraw)
        self.root.bind("<Escape>", lambda _e: self.root.withdraw())

    # ------------------------------------------------------------------
    # UI
    # ------------------------------------------------------------------
    def _build_ui(self):
        tk.Label(
            self.root, text="GESTURE VIRTUAL KEYBOARD",
            font=("Segoe UI", 15, "bold"), bg="#111827", fg="#F9FAFB",
        ).pack(pady=(10, 3))

        tk.Label(
            self.root,
            text="Move the gesture cursor over a key and use your normal left-click gesture.",
            font=("Segoe UI", 9), bg="#111827", fg="#CBD5E1",
        ).pack(pady=(0, 6))

        # Read-only preview strip — never receives keyboard focus.
        self.entry = tk.Entry(
            self.root, textvariable=self.preview,
            font=("Consolas", 14), bg="#F8FAFC", fg="#111827",
            insertbackground="#111827", state="readonly",
        )
        self.entry.pack(fill="x", padx=12, pady=(0, 8))

        rows = [
            list("1234567890") + ["Backspace"],
            list("QWERTYUIOP") + ["Tab"],
            list("ASDFGHJKL")  + ["Enter"],
            list("ZXCVBNM")    + ["Shift", "Caps"],
            ["Space", "Clear", "Capture Target", "Hide"],
        ]

        kb_frame = tk.Frame(self.root, bg="#111827")
        kb_frame.pack(fill="both", expand=True, padx=8)

        self._key_buttons: dict[str, tk.Button] = {}

        for row in rows:
            row_frame = tk.Frame(kb_frame, bg="#111827")
            row_frame.pack(fill="x", pady=2)

            for key in row:
                is_wide = key in (
                    "Backspace", "Enter", "Shift", "Caps", "Tab",
                    "Space", "Clear", "Capture Target", "Hide",
                )
                btn = tk.Button(
                    row_frame, text=key,
                    width=11 if is_wide else 5, height=2,
                    font=("Segoe UI", 9, "bold"),
                    bg="#26364F", fg="#FFFFFF",
                    activebackground="#526FE5", activeforeground="#FFFFFF",
                    relief="flat", bd=0,
                    command=lambda v=key: self.press_key(v),
                )
                btn.pack(side="left", fill="both", expand=True, padx=2)
                btn.bind("<Enter>", lambda _e, b=btn: b.configure(bg="#405B91"))
                btn.bind("<Leave>", lambda _e, b=btn: b.configure(bg="#26364F"))
                self._key_buttons[key] = btn

        self.status = tk.Label(
            self.root,
            text="No target selected — click 'Capture Target' to choose an app.",
            font=("Segoe UI", 9), bg="#111827", fg="#86EFAC",
            wraplength=820,
        )
        self.status.pack(pady=(6, 8))

    # ------------------------------------------------------------------
    # Target capture
    # ------------------------------------------------------------------
    def capture_target(self):
        """Hide the keyboard for 4 s then record the foreground window."""
        if not _WIN32:
            self.status.configure(
                text="pywin32 not installed — cannot target another application."
            )
            return
        if self.target_capture_pending:
            return

        self.target_capture_pending = True
        self.status.configure(text="Switch to your target app within 4 seconds…")
        self.root.after(400,  self.root.withdraw)
        self.root.after(4000, self._finish_capture)

    def _finish_capture(self):
        self.target_capture_pending = False
        try:
            hwnd  = win32gui.GetForegroundWindow()
            title = win32gui.GetWindowText(hwnd).strip()
            print(f"[VK] Capture: hwnd={hwnd}  title={title!r}")

            if hwnd and title and title != "Gesture Virtual Keyboard":
                self.target_hwnd  = hwnd
                self.target_title = title
                self.status.configure(text=f"Target: {title[:70]}")
                print(f"[VK] Target set → hwnd={hwnd}  title={title!r}")
            else:
                self.status.configure(
                    text="No valid target found — try Capture Target again."
                )
                print("[VK] Capture rejected (no title or own window).")
        except Exception as exc:
            self.status.configure(text=f"Capture error: {exc!s:.60}")
            print(f"[VK] Capture exception: {exc}")

        self._raise_overlay()

    # ------------------------------------------------------------------
    # Focus / activation
    # ------------------------------------------------------------------
    def _activate_target(self) -> bool:
        """
        Bring the stored target window to the foreground using
        AttachThreadInput so Windows grants the focus change reliably.
        Returns True when the target is now active.
        """
        if not _WIN32 or self.target_hwnd is None:
            self.status.configure(
                text="No target — click 'Capture Target' first."
            )
            return False

        hwnd = self.target_hwnd

        if not win32gui.IsWindow(hwnd):
            self.target_hwnd = None
            self.target_title = ""
            self.status.configure(text="Target window was closed. Capture a new target.")
            print("[VK] target hwnd is no longer valid.")
            return False

        try:
            # Restore if minimised.
            if win32gui.IsIconic(hwnd):
                win32gui.ShowWindow(hwnd, win32con.SW_RESTORE)

            # Attach our input queue to the target's so SetForegroundWindow works.
            my_tid     = _kernel32.GetCurrentThreadId()
            target_tid = win32process.GetWindowThreadProcessId(hwnd)[0]
            attached   = False

            if my_tid != target_tid:
                attached = bool(
                    _user32.AttachThreadInput(my_tid, target_tid, True)
                )

            _user32.BringWindowToTop(hwnd)
            _user32.SetForegroundWindow(hwnd)

            if attached:
                _user32.AttachThreadInput(my_tid, target_tid, False)

            fg = _user32.GetForegroundWindow()
            success = (fg == hwnd)
            print(
                f"[VK] _activate_target: hwnd={hwnd}  fg_after={fg}"
                f"  match={success}  attached={attached}"
            )
            return True   # proceed even if fg doesn't match immediately

        except Exception as exc:
            self.status.configure(text=f"Activation error: {exc!s:.55}")
            print(f"[VK] _activate_target exception: {exc}")
            return False

    def _raise_overlay(self):
        """Re-assert the keyboard as topmost without grabbing keyboard focus."""
        try:
            self.root.deiconify()
            self.root.attributes("-topmost", True)
            self.root.lift()
        except tk.TclError:
            pass

    # ------------------------------------------------------------------
    # Deferred send helpers
    # ------------------------------------------------------------------
    def _dispatch_char(self, char: str):
        """Activate target then inject a Unicode character after a short delay."""
        if not self._activate_target():
            return
        # 80 ms lets Windows complete the foreground transition before SendInput.
        self.root.after(80, lambda: self._fire_char(char))

    def _fire_char(self, char: str):
        sent = send_unicode_char(char)
        if sent == 2:
            self.status.configure(text=f"Sent: {char!r}  →  {self.target_title[:55]}")
        else:
            self.status.configure(
                text=f"SendInput failed for {char!r} (sent={sent}). "
                     "Try re-capturing the target."
            )
        self.root.after(100, self._raise_overlay)

    def _dispatch_vk(self, name: str):
        """Activate target then inject a virtual-key event after a short delay."""
        vk_code = VK.get(name)
        if vk_code is None:
            print(f"[VK] Unknown VK name: {name!r}")
            return
        if not self._activate_target():
            return
        self.root.after(80, lambda: self._fire_vk(name, vk_code))

    def _fire_vk(self, name: str, vk_code: int):
        sent = send_vk_pair(vk_code)
        if sent == 2:
            self.status.configure(
                text=f"Sent: {name}  →  {self.target_title[:55]}"
            )
        else:
            self.status.configure(
                text=f"SendInput failed for {name!r} (sent={sent})."
            )
        self.root.after(100, self._raise_overlay)

    # ------------------------------------------------------------------
    # Key-press handler
    # ------------------------------------------------------------------
    def press_key(self, key: str):
        now = time.monotonic()
        if now - self.last_key_time < self.key_cooldown:
            return
        self.last_key_time = now

        print(f"[VK] press_key({key!r})  target_hwnd={self.target_hwnd}")

        # --- UI / modifier keys (no SendInput needed) -------------------
        if key == "Capture Target":
            self.capture_target()
            return

        if key == "Hide":
            self.root.withdraw()
            return

        if key == "Clear":
            self.preview.set("")
            self.status.configure(text="Preview cleared.")
            return

        if key == "Shift":
            self.shift_on = not self.shift_on
            self._key_buttons["Shift"].configure(
                bg="#526FE5" if self.shift_on else "#26364F"
            )
            self.status.configure(
                text="Shift " + ("ON" if self.shift_on else "OFF")
            )
            return

        if key == "Caps":
            self.caps_on = not self.caps_on
            self._key_buttons["Caps"].configure(
                bg="#526FE5" if self.caps_on else "#26364F"
            )
            self.status.configure(
                text="Caps Lock " + ("ON" if self.caps_on else "OFF")
            )
            return

        # --- Special keys sent as VK events ------------------------------
        if key == "Backspace":
            self.preview.set(self.preview.get()[:-1])
            self._dispatch_vk("backspace")
            return

        if key == "Enter":
            self.preview.set(self.preview.get() + "\n")
            self._dispatch_vk("enter")
            return

        if key == "Tab":
            self.preview.set(self.preview.get() + "\t")
            self._dispatch_vk("tab")
            return

        if key == "Space":
            self.preview.set(self.preview.get() + " ")
            self._dispatch_char(" ")
            return

        # --- Printable characters ----------------------------------------
        char = key
        if len(key) == 1 and key.isalpha():
            char = key.upper() if (self.shift_on ^ self.caps_on) else key.lower()
        elif self.shift_on and key in "1234567890":
            char = "!@#$%^&*()"["1234567890".index(key)]

        self.preview.set(self.preview.get() + char)
        self._dispatch_char(char)

        if self.shift_on:
            self.shift_on = False
            self._key_buttons["Shift"].configure(bg="#26364F")

    # ------------------------------------------------------------------
    def run(self):
        self.root.mainloop()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main():
    if not _acquire_mutex():
        sys.exit(0)
    try:
        VirtualKeyboard().run()
    finally:
        _release_mutex()


if __name__ == "__main__":
    main()
