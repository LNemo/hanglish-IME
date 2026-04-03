"""macOS implementation of the base hook interface for Hanglish IME."""
from typing import Tuple, Optional
import time
from pynput import keyboard

from .base import BaseHook
from ..core.romanizer import Romanizer
import Quartz


class MacHook(BaseHook):
    # 우리가 의도적으로 생성한 가상 키 이벤트를 식별하기 위한 표식
    SYNTHETIC_MARKER = 12345
    
    # macOS 알파벳 키코드 (A-Z)
    ALPHABET_KEYCODES = {
        0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 11, 12, 13, 14, 15, 16, 17, 
        31, 32, 34, 35, 37, 38, 40, 45, 46
    }

    def __init__(self):
        self.romanizer = Romanizer()
        self.buffer = ""
        # To prevent recursion and leak during simulation
        self.is_simulating = False
        
        # UI callbacks to be set by main app
        self.on_show_candidates = None
        self.on_update_selection = None
        self.on_hide_candidates = None
        
        # Candidate mode state
        self.candidate_mode = False
        self.current_candidates = []
        self.current_selection = 0
        
    def _on_press(self, key):
        if self.is_simulating:
            return True

        # Note: Triggers (Space, Tab, Enter, Numbers) are now handled synchronously 
        # in _darwin_intercept to prevent event leakage. 
        # _on_press only handles buffer updates (Alphabet, Backspace).

        if hasattr(key, 'char') and key.char is not None:
            # Check if it's an alphabet character
            if 'a' <= key.char.lower() <= 'z':
                self.buffer += key.char
                # If we were in candidate mode, update UI
                if self.candidate_mode:
                    self._trigger_candidate_mode()
                
        elif key == keyboard.Key.backspace:
            if self.buffer:
                self.buffer = self.buffer[:-1]
                # If they backspace until buffer is empty, close candidate UI
                if not self.buffer and self.candidate_mode:
                    self._cancel_candidate()
                elif self.candidate_mode:
                    # Update candidate UI dynamically if they delete a letter mid-typing
                    self._trigger_candidate_mode()

    def _trigger_candidate_mode(self):
        if not self.buffer: return

        # NLP 랭킹된 후보 사용
        ranker = self.romanizer._get_ranker()
        dfs_candidates = self.romanizer.get_candidates(self.buffer)

        # 전처리된 입력도 시도 (deureo→deuleo 등)
        preprocessed = self.romanizer._preprocess(self.buffer)
        if preprocessed != self.buffer.lower():
            extra = self.romanizer.get_candidates(preprocessed)
            existing = set(extra)
            for c in dfs_candidates:
                if c not in existing:
                    extra.append(c)
            dfs_candidates = extra

        ranked = ranker.rank(self.buffer, raw_candidates=dfs_candidates if dfs_candidates else None)
        self.current_candidates = [rc.text for rc in ranked[:9]]  # 최대 9개 (숫자키 1-9)

        # 영어 원문이 후보에 없으면 마지막에 추가
        if self.buffer not in self.current_candidates:
            self.current_candidates.append(self.buffer)
            
        if not self.current_candidates: return
        
        # Highlight original english text so it gets replaced on Tab selection or Typing
        # Only do this if we weren't already in candidate mode (to avoid double-highlighting)
        # Set candidate_mode BEFORE highlight to ensure interceptor starts swallowing immediately
        if not self.candidate_mode:
            self.candidate_mode = True
            self._simulate_highlight(len(self.buffer))
        else:
            self.candidate_mode = True
            
        self.current_selection = 0
        if self.on_show_candidates:
            self.on_show_candidates(self.current_candidates, self.current_selection)

    def _nav_candidate(self, delta: int):
        if not self.current_candidates: return
        self.current_selection = (self.current_selection + delta) % len(self.current_candidates)
        if self.on_update_selection:
            self.on_update_selection(self.current_selection)

    def _commit_candidate(self):
        if not self.candidate_mode or not self.current_candidates: return
        
        self.is_simulating = True
        try:
            selected_text = self.current_candidates[self.current_selection]
            buf_len = len(self.buffer)
            
            # Replace the highlighted english buffer with the Korean string
            self.commit_text(selected_text)
            
            # Clear state LAST so injected events and trailing OS events are still intercepted
            self.buffer = ""
            self._cancel_candidate()
        finally:
            self.is_simulating = False

    def _cancel_candidate(self):
        self.candidate_mode = False
        self.current_candidates = []
        self.current_selection = 0
        if self.on_hide_candidates:
            self.on_hide_candidates()
            
        # Deselect text by moving right arrow once if we cancel without committing
        # self._simulate_key(124) # Right Arrow (Optional, depends on exact UX desired)

    def _commit_buffer(self, append_space: bool = False, append_enter: bool = False) -> None:
        """Convert the current buffer, delete it visually if necessary, and commit Hangul."""
        if not self.buffer:
            return
            
        self.is_simulating = True
        try:
            result = self.romanizer.romanize(self.buffer)
            buf_len = len(self.buffer)
            
            # We don't need to highlight and delete here because we already highlight dynamically 
            # when typing or when pressing Tab. However, if they type "annyeong" and press Space, 
            # we still need to highlight it before replacement.
            if not self.candidate_mode:
                 self._simulate_highlight(buf_len)
            
            # Commit translated text (replaces the highlighted text)
            self.commit_text(result)
            
            if append_space:
                self.commit_text(" ")
            if append_enter:
                self._simulate_key(36) # 36 is Return/Enter keycode
            
            # Clear state AFTER all simulated events are posted
            self.buffer = ""
            if self.candidate_mode:
                self._cancel_candidate()
        finally:
            self.is_simulating = False

    def _simulate_highlight(self, count: int):
        # Already inside an is_simulating wrapper from caller
        for _ in range(count):
            # 123 is Left Arrow
            event_down = Quartz.CGEventCreateKeyboardEvent(None, 123, True)
            Quartz.CGEventSetFlags(event_down, Quartz.kCGEventFlagMaskShift)
            Quartz.CGEventSetIntegerValueField(event_down, Quartz.kCGEventSourceUserData, self.SYNTHETIC_MARKER)
            Quartz.CGEventPost(Quartz.kCGHIDEventTap, event_down)
            
            event_up = Quartz.CGEventCreateKeyboardEvent(None, 123, False)
            Quartz.CGEventSetIntegerValueField(event_up, Quartz.kCGEventSourceUserData, self.SYNTHETIC_MARKER)
            Quartz.CGEventPost(Quartz.kCGHIDEventTap, event_up)
            time.sleep(0.005)

    def _simulate_key(self, keycode: int):
        event_down = Quartz.CGEventCreateKeyboardEvent(None, keycode, True)
        Quartz.CGEventSetIntegerValueField(event_down, Quartz.kCGEventSourceUserData, self.SYNTHETIC_MARKER)
        Quartz.CGEventPost(Quartz.kCGHIDEventTap, event_down)
        
        event_up = Quartz.CGEventCreateKeyboardEvent(None, keycode, False)
        Quartz.CGEventSetIntegerValueField(event_up, Quartz.kCGEventSourceUserData, self.SYNTHETIC_MARKER)
        Quartz.CGEventPost(Quartz.kCGHIDEventTap, event_up)
        time.sleep(0.005)

    def commit_text(self, text: str) -> None:
        if not text: return
        
        # Create a special event to post a string
        event = Quartz.CGEventCreateKeyboardEvent(None, 0, True)
        Quartz.CGEventKeyboardSetUnicodeString(event, len(text), text)
        Quartz.CGEventSetIntegerValueField(event, Quartz.kCGEventSourceUserData, self.SYNTHETIC_MARKER)
        Quartz.CGEventPost(Quartz.kCGHIDEventTap, event)
        time.sleep(0.005) # Give OS time to process

    def get_cursor_position(self) -> Tuple[int, int]:
        # Getting actual text cursor coordinate requires macOS Accessibility API via pyobjc AppKit/Quartz.
        # We will implement a mock for now.
        return (0, 0)
        
    def _darwin_intercept(self, event_type, event):
        """Intercepts specific Quartz events from reaching other applications."""
        import Quartz
        
        # 우리가 직접 생성한 키는 무조건 통과 (Marker 확인)
        user_data = Quartz.CGEventGetIntegerValueField(event, Quartz.kCGEventSourceUserData)
        if user_data == self.SYNTHETIC_MARKER:
            return event

        # 시뮬레이션 중에는 다른 모든 '사용자' 입력을 차단하여 잔상 방지
        if self.is_simulating:
            return None

        if event_type in (Quartz.kCGEventKeyDown, Quartz.kCGEventKeyUp):
            keycode = Quartz.CGEventGetIntegerValueField(event, Quartz.kCGKeyboardEventKeycode)
            flags = Quartz.CGEventGetFlags(event)
            
            # 유니코드 문자 입력(0번 keycode)도 통과
            if keycode == 0:
                return event

            # 모디파이어 키 조합 (Cmd, Ctrl, Opt/Alt) 확인
            has_modifier = flags & (Quartz.kCGEventFlagMaskCommand | 
                                   Quartz.kCGEventFlagMaskControl | 
                                   Quartz.kCGEventFlagMaskAlternate)
            
            if has_modifier:
                return event

            # -------------------------------------------------------------------------
            # SYNCHRONOUS TRIGGER HANDLING (핵심: 레이스 컨디션 방지)
            # -------------------------------------------------------------------------
            if event_type == Quartz.kCGEventKeyDown:
                # 후보 모드인 경우
                if self.candidate_mode:
                    if keycode in (48, 125): # Tab (48) or Down (125)
                        self._nav_candidate(1)
                        return None
                    elif keycode == 126: # Up
                        self._nav_candidate(-1)
                        return None
                    elif keycode in (36, 49): # Enter (36) or Space (49)
                        self._commit_candidate()
                        return None
                    elif keycode == 53: # Esc
                        self._cancel_candidate()
                        return None
                    elif 18 <= keycode <= 28: # Numbers 1-9 (Approximately)
                        # More precise mapping for numbers
                        num_map = {18:0, 19:1, 20:2, 21:3, 23:4, 22:5, 26:6, 28:7, 25:8}
                        if keycode in num_map:
                            idx = num_map[keycode]
                            if idx < len(self.current_candidates):
                                self.current_selection = idx
                                self._commit_candidate()
                                return None

                # 일반 버퍼 입력 중인 경우
                elif self.buffer:
                    if keycode == 49: # Space
                        self._commit_buffer(append_space=True)
                        return None
                    elif keycode == 36: # Enter
                        self._commit_buffer(append_enter=True)
                        return None
                    elif keycode == 48: # Tab
                        self._trigger_candidate_mode()
                        return None
                    elif keycode in (123, 124, 125, 126): # Arrows
                        # 버퍼가 있는데 방향키가 들어오면 버퍼를 커밋하고 방향키는 차단(혹은 통과)
                        # 여기선 일단 커밋하고 방향키는 통과시키도록 함 (커서 이동 허용)
                        self._commit_buffer()
                        return event
                    elif keycode in self.ALPHABET_KEYCODES or keycode == 51: # Alphabet or Backspace
                        # 얘네는 _on_press에서 처리하도록 통과시키되, 
                        # intercept 단계에서 whitelist로 명시
                        return event
                    else:
                        # 그 외의 모든 기호 나 키는 버퍼 커밋 후 통과
                        char = self._get_char_from_keycode(keycode)
                        self._commit_buffer()
                        # commit_text(char)를 여기서 해도 되지만, 
                        # 원본 이벤트를 살려두는게(return event) 더 자연스러울 수 있음.
                        return event

            # KeyUp도 눌렸을 때 차단된 키라면 같이 차단해야 함 (잔상 방지)
            if event_type == Quartz.kCGEventKeyUp:
                if self.candidate_mode or self.buffer:
                    # Whitelist (Alphabet, Backspace, Modifiers) 제외하고 차단
                    if keycode in self.ALPHABET_KEYCODES or keycode in (51, 56, 60, 57, 54, 55, 58, 59, 61, 62, 63):
                        return event
                    return None
                    
        return event

    def start(self) -> None:
        print("Starting macOS Keyboard Hook...")
        with keyboard.Listener(on_press=self._on_press, 
                             darwin_intercept=self._darwin_intercept,
                             suppress=True) as listener:
            listener.join()
            
    def start_background(self) -> None:
        print("Starting macOS Keyboard Hook in background...")
        self.listener = keyboard.Listener(on_press=self._on_press, 
                                        darwin_intercept=self._darwin_intercept,
                                        suppress=True)
        self.listener.start()
