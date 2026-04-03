"""Integration tests for IME pipeline using MockHook (no Quartz dependency)."""

import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from hanglish.core.romanizer import Romanizer


class MockHook:
    """MacHook의 핵심 로직을 Quartz 없이 테스트하기 위한 Mock.

    실제 MacHook과 동일한 buffer/candidate/commit 로직을 사용하되,
    Quartz 키보드 이벤트 대신 committed_text에 결과를 저장.
    """

    def __init__(self):
        self.romanizer = Romanizer()
        self.buffer = ""
        self.candidate_mode = False
        self.current_candidates = []
        self.current_selection = 0
        self.committed_text = ""
        self.ui_shown_candidates = None
        self.ui_hidden = False

    def type_char(self, char: str):
        self.buffer += char
        if self.candidate_mode:
            self._trigger_candidate_mode()

    def type_string(self, text: str):
        for ch in text:
            self.type_char(ch)

    def press_backspace(self):
        if self.buffer:
            self.buffer = self.buffer[:-1]
            if not self.buffer and self.candidate_mode:
                self._cancel_candidate()
            elif self.candidate_mode:
                self._trigger_candidate_mode()

    def press_tab(self):
        if self.buffer:
            self._trigger_candidate_mode()

    def press_space(self):
        if self.buffer:
            self._commit_buffer(append_space=True)

    def press_enter_commit(self):
        """Enter in candidate mode — commit selected candidate."""
        if self.candidate_mode:
            self._commit_candidate()

    def press_enter_buffer(self):
        """Enter without candidate mode — commit buffer directly."""
        if self.buffer and not self.candidate_mode:
            self._commit_buffer(append_enter=True)

    def press_down(self):
        self._nav_candidate(1)

    def press_up(self):
        self._nav_candidate(-1)

    def press_escape(self):
        if self.candidate_mode:
            self._cancel_candidate()

    def select_number(self, num: int):
        """숫자키로 후보 직접 선택 (1-indexed)."""
        idx = num - 1
        if self.candidate_mode and idx < len(self.current_candidates):
            self.current_selection = idx
            self._commit_candidate()

    # --- Internal logic (mirrors MacHook) ---

    def _trigger_candidate_mode(self):
        if not self.buffer:
            return

        ranker = self.romanizer._get_ranker()
        dfs_candidates = self.romanizer.get_candidates(self.buffer)

        preprocessed = self.romanizer._preprocess(self.buffer)
        if preprocessed != self.buffer.lower():
            extra = self.romanizer.get_candidates(preprocessed)
            existing = set(extra)
            for c in dfs_candidates:
                if c not in existing:
                    extra.append(c)
            dfs_candidates = extra

        ranked = ranker.rank(self.buffer, raw_candidates=dfs_candidates if dfs_candidates else None)
        self.current_candidates = [rc.text for rc in ranked[:9]]

        if self.buffer not in self.current_candidates:
            self.current_candidates.append(self.buffer)

        if not self.current_candidates:
            return

        self.candidate_mode = True
        self.current_selection = 0
        self.ui_shown_candidates = list(self.current_candidates)
        self.ui_hidden = False

    def _nav_candidate(self, delta: int):
        if not self.current_candidates:
            return
        self.current_selection = (self.current_selection + delta) % len(self.current_candidates)

    def _commit_candidate(self):
        if not self.candidate_mode or not self.current_candidates:
            return
        selected_text = self.current_candidates[self.current_selection]
        self.committed_text += selected_text
        self.buffer = ""
        self._cancel_candidate()

    def _commit_buffer(self, append_space=False, append_enter=False):
        if not self.buffer:
            return
        result = self.romanizer.romanize(self.buffer)
        self.committed_text += result
        if append_space:
            self.committed_text += " "
        if append_enter:
            self.committed_text += "\n"
        self.buffer = ""
        if self.candidate_mode:
            self._cancel_candidate()

    def _cancel_candidate(self):
        self.candidate_mode = False
        self.current_candidates = []
        self.current_selection = 0
        self.ui_hidden = True


@pytest.fixture
def hook():
    return MockHook()


# === Tab → Candidate Mode Tests ===

def test_tab_candidate_annyeong(hook):
    """annyeong 입력 → Tab → 후보 1순위 = 안녕"""
    hook.type_string("annyeong")
    hook.press_tab()
    assert hook.candidate_mode is True
    assert len(hook.current_candidates) > 0
    assert hook.current_candidates[0] == "안녕"


def test_tab_candidate_hello(hook):
    """hello 입력 → Tab → 후보 1순위 = hello (영어 유지)"""
    hook.type_string("hello")
    hook.press_tab()
    assert hook.candidate_mode is True
    assert hook.current_candidates[0] == "hello"


def test_tab_candidate_appleun(hook):
    """Appleun 입력 → Tab → 후보에 Apple은 포함"""
    hook.type_string("Appleun")
    hook.press_tab()
    assert hook.candidate_mode is True
    assert any("Apple" in c and "은" in c for c in hook.current_candidates)


# === Space → Direct Commit Tests ===

def test_space_direct_commit(hook):
    """annyeong 입력 → Space → committed = '안녕 '"""
    hook.type_string("annyeong")
    hook.press_space()
    assert hook.committed_text == "안녕 "
    assert hook.buffer == ""


def test_space_commit_hello(hook):
    """hello 입력 → Space → committed = 'hello '"""
    hook.type_string("hello")
    hook.press_space()
    assert hook.committed_text == "hello "


# === Candidate Navigation Tests ===

def test_candidate_navigation(hook):
    """eobseo 입력 → Tab → Down → Enter → 2번째 후보 커밋"""
    hook.type_string("eobseo")
    hook.press_tab()
    first = hook.current_candidates[0]
    hook.press_down()
    assert hook.current_selection == 1
    hook.press_enter_commit()
    assert hook.committed_text != first  # 2번째 후보가 커밋됨
    assert hook.candidate_mode is False


def test_number_selection(hook):
    """annyeong → Tab → 숫자2 → 2번째 후보 커밋"""
    hook.type_string("annyeong")
    hook.press_tab()
    assert len(hook.current_candidates) >= 2
    second = hook.current_candidates[1]
    hook.select_number(2)
    assert hook.committed_text == second
    assert hook.candidate_mode is False


# === Escape / Backspace Tests ===

def test_escape_cancels(hook):
    """annyeong → Tab → Esc → 후보 모드 종료, buffer 유지"""
    hook.type_string("annyeong")
    hook.press_tab()
    assert hook.candidate_mode is True
    hook.press_escape()
    assert hook.candidate_mode is False
    assert hook.buffer == "annyeong"


def test_backspace_updates_candidates(hook):
    """ann 입력 → Backspace → buffer = 'an'"""
    hook.type_string("ann")
    hook.press_backspace()
    assert hook.buffer == "an"


def test_backspace_in_candidate_mode(hook):
    """annyeong → Tab → Backspace → 후보 갱신"""
    hook.type_string("annyeong")
    hook.press_tab()
    assert hook.candidate_mode is True
    hook.press_backspace()
    assert hook.buffer == "annyeon"
    assert hook.candidate_mode is True  # still in candidate mode


# === Multi-word Sentence Test ===

def test_sentence_workflow(hook):
    """haedang → Space, task → Space 연속 입력"""
    hook.type_string("haedang")
    hook.press_space()
    hook.type_string("task")
    hook.press_space()
    assert "해당" in hook.committed_text
    assert "task" in hook.committed_text


# === Mixed Input Test ===

def test_mixed_hangugeo(hook):
    """hangugeo → Tab → 한국어 1순위"""
    hook.type_string("hangugeo")
    hook.press_tab()
    assert hook.current_candidates[0] == "한국어"


def test_mandeureo_preprocessing(hook):
    """mandeureo → Tab → 만들어 포함"""
    hook.type_string("mandeureo")
    hook.press_tab()
    assert "만들어" in hook.current_candidates
