import sys
import os
from PyQt6.QtWidgets import QApplication
from PyQt6.QtCore import QObject, pyqtSignal, pyqtSlot

# Import UI and Hook
from hanglish.ui.candidate_window import CandidateWindow

class UIBridge(QObject):
    """
    pynput 백그라운드 스레드에서 PyQt6 메인 스레드로 이벤트(후보창 갱신 등)를
    안전하게 전달하기 위한 브릿지 클래스입니다.
    """
    showCandidates = pyqtSignal(list, int)
    updateSelection = pyqtSignal(int)
    hideCandidates = pyqtSignal()

def main():
    print("Hanglish IME Starting...")
    
    # 1. PyQt6 QApplication 초기화 (메인 스레드 점유)
    app = QApplication(sys.argv)
    
    # 2. UI 윈도우 인스턴스 생성
    candidate_window = CandidateWindow()
    # 화면 기준 위치 잡기 로직은 차후 구체화 (MVP에서는 커서 근처 혹은 고정 위치)
    
    # 3. 브릿지 객체 생성 및 시그널 연결
    bridge = UIBridge()
    
    @pyqtSlot(list, int)
    def on_show(candidates, index):
        candidate_window.set_candidates(candidates, index)
        # TODO: 위치 동기화 (현재는 임시로 화면 중앙이나 특정 좌표에 띄우도록 함)
        candidate_window.show()
        
    @pyqtSlot(int)
    def on_update(index):
        candidate_window.update_selection(index)
        
    @pyqtSlot()
    def on_hide():
        candidate_window.hide()

    bridge.showCandidates.connect(on_show)
    bridge.updateSelection.connect(on_update)
    bridge.hideCandidates.connect(on_hide)
    
    # 4. OS별 후킹 모듈 동적 로드 및 백그라운드 실행
    if sys.platform == "darwin":
        from hanglish.platform.macos import MacHook
        hook = MacHook()
    elif sys.platform == "win32":
        from hanglish.platform.windows import WinHook
        # hook = WinHook()
        print("Windows is not fully implemented yet in this MVP.")
        return
    else:
        print("Unsupported OS.")
        return
        
    # 콜백 함수들 연결
    hook.on_show_candidates = bridge.showCandidates.emit
    hook.on_update_selection = bridge.updateSelection.emit
    hook.on_hide_candidates = bridge.hideCandidates.emit

    # 백그라운드 스레드에서 후킹 시작
    hook.start_background()
    print("Background Hook Started. Ready for PyQt6 main loop.")

    # 5. PyQt 메인 이벤트 루프 실행 (블로킹)
    try:
        sys.exit(app.exec())
    except KeyboardInterrupt:
        print("\nHanglish IME Stopped.")

if __name__ == "__main__":
    main()
