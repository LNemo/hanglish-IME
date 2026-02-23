"""
Candidate Window UI for Hanglish IME
"""
from PyQt6.QtWidgets import QWidget, QVBoxLayout, QLabel
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QKeyEvent, QColor
import typing

class CandidateWindow(QWidget):
    # 선택된 후보 문자열을 커밋하기 위해 방출하는 커스텀 시그널
    candidateSelected = pyqtSignal(str)
    # UI가 취소(닫기)되었을 때 알림
    candidateCancelled = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        
        self.candidates: typing.List[str] = []
        self.current_selection = 0
        
        # 윈도우 스타일 설정 (프레임리스, 최상단 유지, 포커스를 뺏지 않도록 ToolTip 설정)
        self.setWindowFlags(
            Qt.WindowType.FramelessWindowHint | 
            Qt.WindowType.WindowStaysOnTopHint |
            Qt.WindowType.ToolTip
        )
        
        # 포커스 뺏김 방지 (매우 중요: 타이핑하던 텍스트 에디터의 포커스를 유지하기 위함)
        self.setAttribute(Qt.WidgetAttribute.WA_ShowWithoutActivating)
        
        # 투명 배경 설정
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        
        # 레이아웃 구성
        self.layout = QVBoxLayout()
        self.layout.setContentsMargins(10, 10, 10, 10)
        self.layout.setSpacing(5)
        self.setLayout(self.layout)
        
        # UI 스타일
        self.setStyleSheet("""
            QWidget {
                background-color: rgba(30, 30, 30, 240);
                border-radius: 8px;
                border: 1px solid #555;
            }
            QLabel {
                color: #EEE;
                font-family: 'Malgun Gothic', 'Apple SD Gothic Neo', sans-serif;
                font-size: 16px;
                padding: 5px 10px;
                border-radius: 4px;
            }
            QLabel[selected="true"] {
                background-color: #3A6EA5;
                color: #FFF;
                font-weight: bold;
            }
        """)

    def set_candidates(self, candidates: typing.List[str], selected_index: int = 0) -> None:
        """후보 리스트를 화면에 표시합니다."""
        self.candidates = candidates
        self.current_selection = selected_index
        
        # 기존 위젯 지우기
        for i in reversed(range(self.layout.count())): 
            widget = self.layout.itemAt(i).widget()
            if widget is not None:
                widget.setParent(None)
                
        # 새 위젯 추가
        for idx, text in enumerate(candidates):
            lbl = QLabel(f"{idx+1}. {text}")
            # 첫 번째 요소 선택 처리
            lbl.setProperty("selected", str(idx == self.current_selection).lower())
            self.layout.addWidget(lbl)
            
        self._update_selection_style()
        self.adjustSize()

    def update_selection(self, selected_index: int):
        self.current_selection = selected_index
        self._update_selection_style()

    def _update_selection_style(self):
        """현재 선택된 항목의 스타일 갱신"""
        for i in range(self.layout.count()):
            widget = self.layout.itemAt(i).widget()
            if isinstance(widget, QLabel):
                widget.setProperty("selected", str(i == self.current_selection).lower())
                # 강제 스타일 업데이트
                widget.style().unpolish(widget)
                widget.style().polish(widget)
