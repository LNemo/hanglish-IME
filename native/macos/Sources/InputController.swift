import Cocoa
import InputMethodKit

@objc(HanglishInputController)
final class HanglishInputController: IMKInputController {
    private var composition = CompositionState()
    private let engine = EngineClient()
    private var task: EngineRequest?
    private var panel: IMKCandidates?
    private var active = false
    private var updatingPanel = false

    deinit { engine.close() }

    override func activateServer(_ sender: Any!) {
        active = true
        super.activateServer(sender)
        // Roman input must not inherit the previously selected Korean key layout.
        (sender as? IMKTextInput)?.overrideKeyboard(withKeyboardNamed: "com.apple.keylayout.US")
    }

    override func deactivateServer(_ sender: Any!) {
        commitComposition(sender)
        active = false
        super.deactivateServer(sender)
    }

    override func inputControllerWillClose() {
        if let client = client() { commit(to: client) }
        task?.cancel()
        composition.reset()
        active = false
        panel?.hide()
        engine.close()
        super.inputControllerWillClose()
    }

    override func handle(_ event: NSEvent!, client sender: Any!) -> Bool {
        guard let event = event, event.type == .keyDown,
              let client = sender as? IMKTextInput else { return false }
        let flags = event.modifierFlags.intersection(.deviceIndependentFlagsMask)
        // Preserve native shortcuts, alternate layout characters and navigation.
        if !flags.intersection([.command, .control, .option]).isEmpty {
            commit(to: client)
            return false
        }
        switch event.keyCode {
        case 36, 76: // Return / keypad Enter: convert, then confirm.
            guard !composition.isEmpty else { return false }
            if flags.contains(.shift) { commit(to: client); return false }
            if !composition.candidates.isEmpty { commit(to: client) }
            else { convert(client: client) }
            return true
        case 48: // Tab / Shift-Tab: convert or cycle candidates.
            guard !composition.isEmpty else { return false }
            if composition.candidates.isEmpty { convert(client: client) }
            else { composition.cycle(delta: flags.contains(.shift) ? -1 : 1); render(client) }
            return true
        case 53: // Escape always retains the original Roman input.
            guard !composition.isEmpty else { return false }
            commit(to: client, raw: true)
            return true
        case 51: // Backspace edits the original source, including after preview.
            guard !composition.isEmpty else { return false }
            task?.cancel()
            _ = composition.backspace()
            render(client)
            return true
        case 125, 126:
            guard !composition.candidates.isEmpty else { commit(to: client); return false }
            composition.cycle(delta: event.keyCode == 125 ? 1 : -1)
            render(client)
            return true
        case 115, 116, 117, 119, 121, 123, 124: // Navigation, forward-delete.
            commit(to: client)
            return false
        default: break
        }
        guard let text = event.characters, !text.isEmpty,
              text.unicodeScalars.allSatisfy({ (0x20...0x7e).contains($0.value) }) else {
            commit(to: client)
            return false
        }
        // Start composition only on an alphanumeric; punctuation still types normally.
        if composition.isEmpty && !text.unicodeScalars.contains(where: {
            CharacterSet.alphanumerics.contains($0)
        }) { return false }
        task?.cancel()
        if !composition.append(text) { commit(to: client); return false }
        render(client)
        return true
    }

    private func convert(client: IMKTextInput) {
        guard let request = composition.beginConversion() else { return }
        let id = UUID().uuidString
        task = engine.convert(request.input, requestID: id) { [weak self, weak client] result in
            guard let self = self, self.active, let client = client else { return }
            switch result {
            case .success(let reply):
                guard self.composition.accept(candidates: reply.candidates.map(\.text), revision: request.revision) else { return }
            case .failure(let error):
                guard self.composition.fail(message: error.localizedDescription, revision: request.revision) else { return }
                NSSound.beep()
            }
            self.task = nil
            self.render(client)
        }
    }

    private func render(_ client: IMKTextInput) {
        let value = composition.displayText
        client.setMarkedText(value, selectionRange: NSRange(location: value.utf16.count, length: 0),
                             replacementRange: NSRange(location: NSNotFound, length: 0))
        if composition.candidates.isEmpty { panel?.hide(); return }
        if panel == nil {
            panel = IMKCandidates(server: server(), panelType: kIMKSingleColumnScrollingCandidatePanel)
            // Key events reach this controller first, so digits remain editable text.
            panel?.setAttributes([IMKCandidatesSendServerKeyEventFirst: true])
        }
        updatingPanel = true
        panel?.update()
        if let panel = panel {
            _ = panel.selectCandidate(withIdentifier: panel.candidateStringIdentifier(value))
        }
        panel?.show(kIMKLocateCandidatesBelowHint)
        updatingPanel = false
    }

    private func commit(to client: IMKTextInput, raw: Bool = false) {
        task?.cancel()
        task = nil
        let text = composition.finish(raw: raw)
        panel?.hide()
        if !text.isEmpty {
            client.insertText(text, replacementRange: NSRange(location: NSNotFound, length: 0))
        }
    }

    override func commitComposition(_ sender: Any!) {
        if let client = sender as? IMKTextInput { commit(to: client) }
        else if let client = client() { commit(to: client) }
    }

    override func candidates(_ sender: Any!) -> [Any]! { composition.candidates }
    override func composedString(_ sender: Any!) -> Any! { composition.displayText }
    override func originalString(_ sender: Any!) -> NSAttributedString! { NSAttributedString(string: composition.raw) }

    override func candidateSelected(_ candidateString: NSAttributedString!) {
        guard let candidateString = candidateString, composition.select(text: candidateString.string),
              let client = client() else { return }
        commit(to: client)
    }

    override func candidateSelectionChanged(_ candidateString: NSAttributedString!) {
        // Native candidate selection preview. Do not recursively rebuild its panel.
        guard !updatingPanel, let candidateString = candidateString, composition.select(text: candidateString.string),
              let client = client() else { return }
        let text = composition.displayText
        client.setMarkedText(text, selectionRange: NSRange(location: text.utf16.count, length: 0),
                             replacementRange: NSRange(location: NSNotFound, length: 0))
    }

    override func menu() -> NSMenu! {
        let menu = NSMenu(title: "Hanglish")
        let status = composition.errorMessage ?? (composition.pendingRevision != nil
            ? localized("Converting… You can keep editing the original text")
            : localized("Enter convert → Enter confirm · Tab candidates · Esc original"))
        let item = NSMenuItem(title: status, action: nil, keyEquivalent: "")
        item.isEnabled = false
        menu.addItem(item)
        return menu
    }
}
