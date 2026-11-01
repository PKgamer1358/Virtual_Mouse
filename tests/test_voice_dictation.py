r"""
Tests for voice_assistant.py – dictation pipeline and command routing.

Run from the project root with:

    .\.venv311\Scripts\python.exe -m pytest tests\test_voice_dictation.py -v

All GUI and speech-recognition operations are mocked so no real microphone,
clipboard, or Notepad window is needed.
"""


import queue
import sys
import time
import threading
import types
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch, call

# ---------------------------------------------------------------------------
# Ensure src/ is importable without installing the package
# ---------------------------------------------------------------------------
SRC = Path(__file__).resolve().parent.parent / "src"
sys.path.insert(0, str(SRC))

# ---------------------------------------------------------------------------
# Stub heavy optional dependencies so the module imports cleanly in CI
# ---------------------------------------------------------------------------
def _stub(name):
    mod = types.ModuleType(name)
    sys.modules[name] = mod
    return mod


for _name in ["pyttsx3", "sounddevice", "pyautogui"]:
    if _name not in sys.modules:
        _stub(_name)

# pyttsx3
sys.modules["pyttsx3"].init = MagicMock(return_value=MagicMock())  # type: ignore

# sounddevice
sys.modules["sounddevice"].RawInputStream = MagicMock()  # type: ignore

# pyautogui
_pag = sys.modules["pyautogui"]
for _attr in ("hotkey", "press", "write", "click",
              "rightClick", "doubleClick", "scroll", "FAILSAFE"):
    setattr(_pag, _attr, MagicMock())

# win32clipboard / win32con stubs (voice_assistant uses these from a bg thread)
_win32_clipboard: list[str] = []   # records what was placed on the clipboard

_win32cb = _stub("win32clipboard")
_win32cb.OpenClipboard = MagicMock()          # type: ignore
_win32cb.EmptyClipboard = MagicMock()         # type: ignore
_win32cb.CloseClipboard = MagicMock()         # type: ignore
_win32cb.SetClipboardData = MagicMock(         # type: ignore
    side_effect=lambda fmt, text: _win32_clipboard.append(text)
)

_win32con = _stub("win32con")
_win32con.CF_UNICODETEXT = 13                  # type: ignore

# assemblyai stubs
if "assemblyai" not in sys.modules:
    _aai = _stub("assemblyai")
    _aai_s = _stub("assemblyai.streaming")
    _aai_v3 = _stub("assemblyai.streaming.v3")
    import enum as _enum

    class _RE(_enum.Enum):
        Turn = "Turn"
        Error = "Error"

    _aai_v3.RealTimeEvents = _RE              # type: ignore
    _aai_v3.RealTimeParameters = MagicMock()  # type: ignore
    _aai_v3.RealTimeTranscriber = MagicMock() # type: ignore
    _aai_v3.RealTimeTranscriberOptions = MagicMock()  # type: ignore

# voice_stats stubs
if "voice_stats" not in sys.modules:
    _vs = _stub("voice_stats")
    for _fn in ("record_recognition", "record_command",
                "record_ignored", "record_dictation", "record_error"):
        setattr(_vs, _fn, MagicMock())

# ---------------------------------------------------------------------------
# Import module under test
# ---------------------------------------------------------------------------
# pyrefly: ignore[missing-import]
import voice_assistant as va  # noqa: E402


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_turn(turn_order: int, transcript: str, end_of_turn: bool = True):
    """Return a minimal TurnEvent-like object."""
    evt = MagicMock()
    evt.turn_order = turn_order
    evt.transcript = transcript
    evt.end_of_turn = end_of_turn
    return evt


def _make_assistant(speak=False) -> va.VoiceAssistant:
    """Return a VoiceAssistant that is ready to process turns (running=True)."""
    asst = va.VoiceAssistant(speak=speak)
    # Tests drive _handle_turn_event directly; running=True prevents the
    # early-return guard (`not self.running`) from blocking them.
    asst.running = True
    return asst


def _track_insertion(asst: va.VoiceAssistant):
    """Monkey-patch the insertion queue and return a list that collects puts."""
    inserted: list[str] = []
    original_put = asst._insertion_queue.put

    def track_put(text):
        inserted.append(text)
        original_put(text)

    asst._insertion_queue.put = track_put
    return inserted


# ---------------------------------------------------------------------------
# Test: _normalize
# ---------------------------------------------------------------------------

class TestNormalize(unittest.TestCase):
    def test_strips_trailing_punctuation(self):
        self.assertEqual(va.VoiceAssistant._normalize("Hello."), "hello")

    def test_collapses_whitespace(self):
        self.assertEqual(va.VoiceAssistant._normalize("  stop  dictation  "), "stop dictation")

    def test_lowercase(self):
        self.assertEqual(va.VoiceAssistant._normalize("STOP DICTATION"), "stop dictation")

    def test_preserves_internal_commas(self):
        # commas *inside* a phrase are not surrounding punctuation and are kept
        result = va.VoiceAssistant._normalize("one, two, three")
        self.assertEqual(result, "one, two, three")


# ---------------------------------------------------------------------------
# Test 1 & 2: turn_order deduplication
# ---------------------------------------------------------------------------

class TestTurnOrderDeduplication(unittest.TestCase):
    def setUp(self):
        self.asst = _make_assistant()
        self.asst.dictation_mode = True
        self.inserted = _track_insertion(self.asst)

    # Test 1 – two consecutive DIFFERENT phrases → both inserted
    def test_two_different_phrases_both_inserted(self):
        self.asst._handle_turn_event(_make_turn(1, "Hello, my name is Pranav."))
        self.asst._handle_turn_event(_make_turn(2, "I study at RV College."))
        self.assertEqual(len(self.inserted), 2)
        self.assertIn("Hello, my name is Pranav.", self.inserted)
        self.assertIn("I study at RV College.", self.inserted)

    # Test 2 – same turn_order delivered twice → inserted only once
    def test_same_turn_order_inserted_once(self):
        evt = _make_turn(5, "duplicate phrase")
        self.asst._handle_turn_event(evt)
        self.asst._handle_turn_event(evt)   # re-delivery of same turn_order
        self.assertEqual(len(self.inserted), 1)

    # Partial events (end_of_turn=False) must be skipped entirely
    def test_partial_event_skipped(self):
        self.asst._handle_turn_event(_make_turn(3, "partial text", end_of_turn=False))
        self.assertEqual(len(self.inserted), 0)


# ---------------------------------------------------------------------------
# Tests 3, 4, 5: legitimate repetitions must NOT be filtered
# ---------------------------------------------------------------------------

class TestLegitimateRepetitions(unittest.TestCase):
    def setUp(self):
        self.asst = _make_assistant()
        self.asst.dictation_mode = True
        self.inserted = _track_insertion(self.asst)

    # Test 3 – repeated word inside a single turn: "very, very good"
    def test_repeated_word_in_phrase_inserted(self):
        self.asst._handle_turn_event(_make_turn(10, "very, very good"))
        self.assertEqual(len(self.inserted), 1)
        self.assertIn("very, very good", self.inserted[0])

    # Test 4 – identical phrase spoken twice, with gap > _min_repeat_gap
    def test_repeated_sentence_separate_turns_both_inserted(self):
        self.asst._handle_turn_event(_make_turn(11, "I am happy."))
        # Wind back the timestamp so the text guard allows the repeat
        self.asst._last_committed_time -= (self.asst._min_repeat_gap + 1)
        self.asst._handle_turn_event(_make_turn(12, "I am happy."))
        self.assertEqual(len(self.inserted), 2,
                         "Repeated sentence after gap must be inserted twice")

    # Test 5 – repeated number sequence: "one, two, three" × 2
    def test_repeated_number_sequence(self):
        self.asst._handle_turn_event(_make_turn(13, "one, two, three"))
        self.asst._last_committed_time -= (self.asst._min_repeat_gap + 1)
        self.asst._handle_turn_event(_make_turn(14, "one, two, three"))
        self.assertEqual(len(self.inserted), 2)


# ---------------------------------------------------------------------------
# Test 6: two finalized turns arriving almost simultaneously
# ---------------------------------------------------------------------------

class TestSimultaneousArrivals(unittest.TestCase):
    def test_two_turns_same_time(self):
        asst = _make_assistant()
        asst.dictation_mode = True
        inserted = _track_insertion(asst)

        results: list[Exception | None] = [None, None]

        def fire(turn_order, text, slot):
            try:
                asst._handle_turn_event(_make_turn(turn_order, text))
            except Exception as exc:
                results[slot] = exc

        t1 = threading.Thread(target=fire, args=(20, "First segment.", 0))
        t2 = threading.Thread(target=fire, args=(21, "Second segment.", 1))
        t1.start(); t2.start()
        t1.join(); t2.join()

        self.assertIsNone(results[0])
        self.assertIsNone(results[1])
        self.assertEqual(len(inserted), 2)


# ---------------------------------------------------------------------------
# Test 7: partial then corrected final for the same turn_order
# ---------------------------------------------------------------------------

class TestPartialThenFinal(unittest.TestCase):
    def test_only_final_is_inserted(self):
        asst = _make_assistant()
        asst.dictation_mode = True
        inserted: list[str] = []
        asst._insertion_queue.put = lambda t: inserted.append(t)

        # Partial (end_of_turn=False) must be silently ignored
        asst._handle_turn_event(_make_turn(30, "helo world", end_of_turn=False))
        # Final (end_of_turn=True) for the same turn_order must be inserted
        asst._handle_turn_event(_make_turn(30, "hello world", end_of_turn=True))

        self.assertEqual(len(inserted), 1)
        self.assertEqual(inserted[0], "hello world")


# ---------------------------------------------------------------------------
# Test 8: punctuation is preserved in inserted text
# ---------------------------------------------------------------------------

class TestPunctuationPreservation(unittest.TestCase):
    def test_punctuation_preserved_in_insertion(self):
        asst = _make_assistant()
        asst.dictation_mode = True
        inserted: list[str] = []
        asst._insertion_queue.put = lambda t: inserted.append(t)

        raw = "Hello, my name is Pranav. I study at RV College."
        asst._handle_turn_event(_make_turn(40, raw))

        self.assertEqual(len(inserted), 1)
        # Raw text (with commas, periods, capitalisation) must reach the queue
        self.assertEqual(inserted[0], raw)

    def test_no_backspace_pressed(self):
        """_type_text must never send a Backspace keystroke."""
        asst = _make_assistant()
        press_mock = sys.modules["pyautogui"].press
        press_mock.reset_mock()
        with patch.object(sys.modules["pyautogui"], "hotkey"):
            asst._type_text("Hello.")
        # pyautogui.press should NOT have been called with 'backspace'
        backspace_calls = [
            c for c in press_mock.call_args_list
            if c.args and c.args[0] == "backspace"
        ]
        self.assertEqual(backspace_calls, [])

    def test_trailing_space_added(self):
        """_type_text appends a trailing space to separate consecutive turns."""
        asst = _make_assistant()
        _win32_clipboard.clear()
        with patch.object(sys.modules["pyautogui"], "hotkey"):
            asst._type_text("Hello.")
        # The win32clipboard stub records what SetClipboardData received.
        self.assertTrue(len(_win32_clipboard) > 0, "Clipboard must have been written")
        self.assertTrue(
            _win32_clipboard[-1].endswith(" "),
            "Trailing space must be present in clipboard content",
        )


# ---------------------------------------------------------------------------
# Test 9: clipboard failure → keyboard fallback, no duplication
# ---------------------------------------------------------------------------

class TestClipboardFallback(unittest.TestCase):
    def test_keyboard_fallback_when_clipboard_path_raises(self):
        """When pyautogui.hotkey raises, _type_text falls back to write."""
        asst = _make_assistant()

        write_mock = sys.modules["pyautogui"].write
        write_mock.reset_mock()

        # Make ctrl+v raise so pasted stays False and the fallback is triggered.
        with patch.object(sys.modules["pyautogui"], "hotkey",
                          side_effect=RuntimeError("hotkey unavailable")):
            asst._type_text("test text")

        write_mock.assert_called_once()
        args = write_mock.call_args[0]
        self.assertTrue(args[0].startswith("test text"),
                        "Fallback must write the original text")

    def test_no_double_insertion_on_fallback(self):
        """When clipboard succeeds (win32clipboard stub), write must not be called."""
        asst = _make_assistant()
        write_mock = sys.modules["pyautogui"].write
        write_mock.reset_mock()
        with patch.object(sys.modules["pyautogui"], "hotkey"):
            asst._type_text("clipboard ok")
        write_mock.assert_not_called()


# ---------------------------------------------------------------------------
# Test 10: start/stop dictation cycles
# ---------------------------------------------------------------------------

class TestStartStopDictation(unittest.TestCase):
    def test_start_stop_cycle(self):
        asst = _make_assistant()
        asst.say = MagicMock()

        self.assertFalse(asst.dictation_mode)

        asst.handle_text("start dictation")
        self.assertTrue(asst.dictation_mode)

        asst.handle_text("stop dictation")
        self.assertFalse(asst.dictation_mode)

        # Second cycle
        asst.handle_text("begin dictation")
        self.assertTrue(asst.dictation_mode)

        asst.handle_text("end dictation")
        self.assertFalse(asst.dictation_mode)

    def test_dictation_command_text_not_inserted(self):
        """start/stop dictation keywords must never be typed into the target app."""
        asst = _make_assistant()
        asst.say = MagicMock()
        inserted: list[str] = []
        asst._insertion_queue.put = lambda t: inserted.append(t)

        asst.handle_text("start dictation")
        asst.handle_text("stop dictation")
        self.assertEqual(inserted, [])


# ---------------------------------------------------------------------------
# Test 11: voice commands outside dictation
# ---------------------------------------------------------------------------

class TestCommandsOutsideDictation(unittest.TestCase):
    def setUp(self):
        self.asst = _make_assistant()
        self.asst.say = MagicMock()
        self.asst.dictation_mode = False

    def test_select_all_fires_hotkey(self):
        with patch.object(sys.modules["pyautogui"], "hotkey") as mock:
            self.asst.handle_text("select all")
        mock.assert_called_once_with("ctrl", "a")

    def test_press_enter_fires_press(self):
        with patch.object(sys.modules["pyautogui"], "press") as mock:
            self.asst.handle_text("press enter")
        mock.assert_called_once_with("enter")

    def test_unmapped_phrase_does_not_raise(self):
        # Should record_ignored and print, not raise
        self.asst.handle_text("something completely unmapped")


# ---------------------------------------------------------------------------
# Test 12: voice commands while dictation IS active
# ---------------------------------------------------------------------------

class TestCommandsDuringDictation(unittest.TestCase):
    def setUp(self):
        self.asst = _make_assistant()
        self.asst.say = MagicMock()
        self.asst.dictation_mode = True
        self.inserted: list[str] = []
        self.asst._insertion_queue.put = lambda t: self.inserted.append(t)

    def test_stop_dictation_command_not_typed(self):
        self.asst.handle_text("stop dictation")
        self.assertFalse(self.asst.dictation_mode)
        self.assertEqual(self.inserted, [])

    def test_ordinary_speech_typed_during_dictation(self):
        self.asst.handle_text("Hello world")
        self.assertEqual(len(self.inserted), 1)
        self.assertEqual(self.inserted[0], "Hello world")

    def test_new_line_presses_enter_not_typed(self):
        with patch.object(sys.modules["pyautogui"], "press") as mock:
            self.asst.handle_text("new line")
        mock.assert_called_once_with("enter")
        self.assertEqual(self.inserted, [])

    def test_voice_help_not_typed(self):
        self.asst.handle_text("voice help")
        self.assertEqual(self.inserted, [])


# ---------------------------------------------------------------------------
# Insertion order test
# ---------------------------------------------------------------------------

class TestInsertionOrder(unittest.TestCase):
    def test_insertion_order_preserved(self):
        asst = _make_assistant()
        typed: list[str] = []

        def fake_type(text):
            typed.append(text)

        asst._type_text = fake_type

        # Start the insertion worker with running=True
        worker = threading.Thread(target=asst._insertion_worker, daemon=True)
        worker.start()

        phrases = ["First.", "Second.", "Third."]
        for p in phrases:
            asst._insertion_queue.put(p)

        asst._insertion_queue.join()
        asst.running = False
        worker.join(timeout=2)

        self.assertEqual(typed, phrases)


# ---------------------------------------------------------------------------
# Echo suppression tests
# ---------------------------------------------------------------------------

class TestEchoSuppression(unittest.TestCase):
    def test_recognition_suppressed_during_speech(self):
        asst = _make_assistant()
        asst.dictation_mode = True
        inserted: list[str] = []
        asst._insertion_queue.put = lambda t: inserted.append(t)

        asst._ignore_recognition_until = time.monotonic() + 30.0  # far future

        asst._handle_turn_event(_make_turn(50, "this should be suppressed"))
        self.assertEqual(inserted, [])

    def test_recognition_accepted_after_speech_ends(self):
        asst = _make_assistant()
        asst.dictation_mode = True
        inserted: list[str] = []
        asst._insertion_queue.put = lambda t: inserted.append(t)

        asst._ignore_recognition_until = time.monotonic() - 1.0  # already past

        asst._handle_turn_event(_make_turn(51, "this should be accepted"))
        self.assertEqual(len(inserted), 1)


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    unittest.main(verbosity=2)
