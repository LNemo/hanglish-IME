import Foundation
import Darwin

@main
struct EngineClientTests {
    private final class Box<T> { var value: T? }
    // Stored weak references work with both older and newer Swift compilers.
    private final class WeakRequest {
        weak var value: EngineRequest?
        init(_ value: EngineRequest?) { self.value = value }
    }
    private static var failures: [String] = []

    static func main() throws {
        let fixture = try makeFixture()
        defer { try? FileManager.default.removeItem(at: fixture.directory) }

        testSuccess(fixture)
        testInvalidResponse(fixture)
        testOversizedResponse(fixture)
        testMismatchedID(fixture)
        testReportedError(fixture)
        testRefusesOverflow(fixture)
        testCurrentAndQueuedCancellation(fixture)
        testCancellationReleasesRequests(fixture)
        testTimeoutThenRestart(fixture)
        testUncooperativeChildIsReaped(fixture)
        testCrashThenRestart(fixture)

        if failures.isEmpty {
            print("EngineClientTests: all process, protocol, timeout, cancellation, and restart tests passed")
        } else {
            let output = failures.map { "FAIL: \($0)" }.joined(separator: "\n") + "\n"
            FileHandle.standardError.write(Data(output.utf8))
            exit(1)
        }
    }

    private struct Fixture {
        let directory: URL
        let script: URL
        let stubbornPID: URL
    }

    private static func makeFixture() throws -> Fixture {
        let directory = FileManager.default.temporaryDirectory
            .appendingPathComponent("hanglish-engine-tests-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        let script = directory.appendingPathComponent("fixture.sh")
        let stubbornPID = directory.appendingPathComponent("stubborn.pid")
        let source = #"""
#!/bin/sh
while IFS= read -r line; do
  case "$line" in *'"context":""'*) : ;; *) exit 91 ;; esac
  case "$line" in *'"engine":"neural"'*) : ;; *) exit 92 ;; esac
  case "$line" in *'"mode":"batch"'*) : ;; *) exit 93 ;; esac
  case "$line" in *'"raw":false'*) : ;; *) exit 94 ;; esac
  case "$line" in
    *'"input":"invalid"'*) printf 'not-json\n' ;;
    *'"input":"oversize"'*) printf '{"result":"xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx","candidates":[{"text":"xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"}],"version":"fixture-v1","request_id":"oversized-id"}\n' ;;
    *'"input":"mismatch"'*) printf '{"result":"wrong","candidates":[{"text":"wrong"}],"version":"fixture-v1","request_id":"wrong-id"}\n' ;;
    *'"input":"reported-error"'*) printf '{"error":"internal detail must not escape","request_id":"error-id"}\n' ;;
    *'"input":"crash"'*) exit 17 ;;
    *'"input":"timeout"'*) sleep 2; printf '{"result":"late","candidates":[{"text":"late"}],"version":"fixture-v1","request_id":"timeout-id"}\n' ;;
    *'"input":"stubborn"'*) printf '%s\n' "$$" > "$1"; trap '' TERM; while :; do :; done ;;
    *'"input":"slow-current"'*) sleep 0.25; printf '{"result":"current","candidates":[{"text":"current"}],"version":"fixture-v1","request_id":"current-id"}\n' ;;
    *'"input":"slow-overflow"'*) sleep 0.35; printf '{"result":"overflow","candidates":[{"text":"overflow"}],"version":"fixture-v1","request_id":"overflow-1"}\n' ;;
    *'"input":"queued-cancel"'*) printf '{"result":"queued","candidates":[{"text":"queued"}],"version":"fixture-v1","request_id":"queued-id"}\n' ;;
    *'"request_id":"overflow-2"'*) printf '{"result":"second","candidates":[{"text":"second"}],"version":"fixture-v1","request_id":"overflow-2"}\n' ;;
    *'"request_id":"restart-id"'*) printf '{"result":"restarted","candidates":[{"text":"restarted"}],"version":"fixture-v1","request_id":"restart-id"}\n' ;;
    *'"request_id":"ok-id"'*) printf '{"result":"입니다","candidates":[{"text":"입니다"},{"text":"임니다"}],"version":"fixture-v1","request_id":"ok-id"}\n' ;;
    *) printf '{"error":"unknown fixture request","request_id":"unknown"}\n' ;;
  esac
done
"""#
        try Data(source.utf8).write(to: script)
        try FileManager.default.setAttributes([.posixPermissions: 0o755], ofItemAtPath: script.path)
        return Fixture(directory: directory, script: script, stubbornPID: stubbornPID)
    }

    private static func client(_ fixture: Fixture, timeout: TimeInterval = 1,
                               maxPending: Int = 8,
                               maxFrameBytes: Int = 1_048_576) -> EngineClient {
        EngineClient(executableURL: URL(fileURLWithPath: "/bin/sh"),
                     arguments: [fixture.script.path, fixture.stubbornPID.path],
                     currentDirectoryURL: fixture.directory,
                     requestTimeout: timeout, idleTimeout: 0.15,
                     maxPending: maxPending, maxFrameBytes: maxFrameBytes)
    }

    private static func testSuccess(_ fixture: Fixture) {
        let result = request(client(fixture), input: "imnida", id: "ok-id")
        guard case .success(let reply)? = result else {
            failures.append("success: expected a valid reply")
            return
        }
        expect(reply.result == "입니다", "success: wrong first result")
        expect(reply.candidates.map(\.text) == ["입니다", "임니다"], "success: candidates changed")
        expect(reply.version == "fixture-v1", "success: version changed")
    }

    private static func testInvalidResponse(_ fixture: Fixture) {
        expect(error(from: request(client(fixture), input: "invalid", id: "invalid-id")) == .invalidResponse,
               "invalid: malformed JSON was not rejected")
    }

    private static func testOversizedResponse(_ fixture: Fixture) {
        let limited = client(fixture, maxFrameBytes: 128)
        expect(error(from: request(limited, input: "oversize", id: "oversized-id")) == .invalidResponse,
               "frame limit: oversized response was not rejected")
    }

    private static func testMismatchedID(_ fixture: Fixture) {
        expect(error(from: request(client(fixture), input: "mismatch", id: "expected-id")) == .invalidResponse,
               "id mismatch: response was not rejected")
    }

    private static func testReportedError(_ fixture: Fixture) {
        expect(error(from: request(client(fixture), input: "reported-error", id: "error-id")) == .unavailable,
               "reported error: internal child error was not reduced to a safe failure")
    }

    private static func testRefusesOverflow(_ fixture: Fixture) {
        let client = client(fixture, maxPending: 2)
        let first = Box<Result<ConversionReply, Error>>()
        let second = Box<Result<ConversionReply, Error>>()
        let third = Box<Result<ConversionReply, Error>>()
        let firstRequest = client.convert("slow-overflow", requestID: "overflow-1") { first.value = $0 }
        let secondRequest = client.convert("second", requestID: "overflow-2") { second.value = $0 }
        let thirdRequest = client.convert("third", requestID: "overflow-3") { third.value = $0 }
        _ = wait(until: { third.value != nil }, timeout: 1)
        expect(error(from: third.value) == .busy, "overflow: ninth-style pending request was not refused")
        firstRequest.cancel()
        secondRequest.cancel()
        thirdRequest.cancel()
        client.close()
    }

    private static func testCurrentAndQueuedCancellation(_ fixture: Fixture) {
        let client = client(fixture)
        var currentCalled = false
        var queuedCalled = false
        let final = Box<Result<ConversionReply, Error>>()
        let current = client.convert("slow-current", requestID: "current-id") { _ in currentCalled = true }
        let queued = client.convert("queued-cancel", requestID: "queued-id") { _ in queuedCalled = true }
        let surviving = client.convert("restart", requestID: "restart-id") { final.value = $0 }
        _ = RunLoop.main.run(mode: .default, before: Date(timeIntervalSinceNow: 0.05))
        current.cancel()
        queued.cancel()
        expect(wait(until: { final.value != nil }, timeout: 2), "cancellation: surviving queued request timed out")
        expect(!currentCalled, "cancellation: cancelled current request completed")
        expect(!queuedCalled, "cancellation: cancelled queued request completed")
        if case .success(let reply)? = final.value {
            expect(reply.result == "restarted", "cancellation: surviving request was corrupted")
        } else {
            failures.append("cancellation: surviving request failed")
        }
        surviving.cancel()
        client.close()
    }

    private static func testCancellationReleasesRequests(_ fixture: Fixture) {
        let client = client(fixture)
        var current: EngineRequest? = client.convert("slow-current", requestID: "current-id") { _ in }
        var queued: EngineRequest? = client.convert("queued-cancel", requestID: "queued-id") { _ in }
        let releasedCurrent = WeakRequest(current)
        let releasedQueued = WeakRequest(queued)
        current?.cancel()
        queued?.cancel()
        current = nil
        queued = nil
        expect(releasedCurrent.value == nil && releasedQueued.value == nil,
               "cancellation: client retained cancelled request handles")
        client.close()
    }

    private static func testTimeoutThenRestart(_ fixture: Fixture) {
        // The fixture stalls for two seconds. Allow normal process scheduling
        // after a busy signing build without requiring restart within 150 ms.
        let client = client(fixture, timeout: 1)
        defer { client.close() }
        expect(error(from: request(client, input: "timeout", id: "timeout-id", timeout: 3)) == .timedOut,
               "timeout: request did not time out")
        let restarted = request(client, input: "restart", id: "restart-id", timeout: 3)
        guard case .success(let reply)? = restarted else {
            failures.append("timeout: next request failed: \(String(describing: restarted))")
            return
        }
        expect(reply.result == "restarted", "timeout: restarted process returned wrong data")
    }

    private static func testUncooperativeChildIsReaped(_ fixture: Fixture) {
        try? FileManager.default.removeItem(at: fixture.stubbornPID)
        // This child never replies. Keep a bounded timeout while giving the
        // following healthy process the same scheduling budget as other tests.
        let client = client(fixture, timeout: 1)
        defer { client.close() }
        expect(error(from: request(client, input: "stubborn", id: "stubborn-id", timeout: 3)) == .timedOut,
               "reaping: stubborn child did not time out")
        guard let data = try? Data(contentsOf: fixture.stubbornPID),
              let text = String(data: data, encoding: .utf8),
              let pid = Int32(text.trimmingCharacters(in: .whitespacesAndNewlines)) else {
            failures.append("reaping: fixture did not record its process ID")
            return
        }
        expect(wait(until: { Darwin.kill(pid, 0) == -1 && errno == ESRCH }, timeout: 2),
               "reaping: child that ignored termination remained alive")
        let restarted = request(client, input: "restart", id: "restart-id", timeout: 3)
        guard case .success(let reply)? = restarted else {
            failures.append("reaping: next request failed: \(String(describing: restarted))")
            return
        }
        expect(reply.result == "restarted", "reaping: restarted process returned wrong data")
    }

    private static func testCrashThenRestart(_ fixture: Fixture) {
        let client = client(fixture)
        let crashed = Box<Result<ConversionReply, Error>>()
        let pending = Box<Result<ConversionReply, Error>>()
        let crashRequest = client.convert("crash", requestID: "crash-id") { crashed.value = $0 }
        let pendingRequest = client.convert("restart", requestID: "restart-id") { pending.value = $0 }
        expect(wait(until: { crashed.value != nil && pending.value != nil }, timeout: 2),
               "crash: process EOF did not finish every pending request")
        expect(error(from: crashed.value) == .unavailable,
               "crash: active request did not receive an unavailable error")
        expect(error(from: pending.value) == .unavailable,
               "crash: queued request did not receive an unavailable error")
        withExtendedLifetime((crashRequest, pendingRequest)) {}
        guard case .success? = request(client, input: "restart", id: "restart-id", timeout: 2) else {
            failures.append("crash: process did not restart")
            return
        }
        client.close()
    }

    private static func request(_ client: EngineClient, input: String, id: String,
                                timeout: TimeInterval = 2) -> Result<ConversionReply, Error>? {
        let box = Box<Result<ConversionReply, Error>>()
        let handle = client.convert(input, requestID: id) { box.value = $0 }
        _ = wait(until: { box.value != nil }, timeout: timeout)
        withExtendedLifetime(handle) {}
        return box.value
    }

    private static func wait(until condition: () -> Bool, timeout: TimeInterval) -> Bool {
        let deadline = Date(timeIntervalSinceNow: timeout)
        while !condition(), Date() < deadline {
            _ = RunLoop.main.run(mode: .default, before: Date(timeIntervalSinceNow: 0.01))
        }
        return condition()
    }

    private static func error(from result: Result<ConversionReply, Error>?) -> EngineError? {
        guard case .failure(let error)? = result else { return nil }
        return error as? EngineError
    }

    private static func expect(_ condition: @autoclosure () -> Bool, _ message: String) {
        if !condition() { failures.append(message) }
    }
}
