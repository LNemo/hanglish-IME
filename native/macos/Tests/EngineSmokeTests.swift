import Foundation

@main
struct EngineSmokeTests {
    private struct Fixture {
        let id: String
        let input: String
        let expected: String
    }

    private static let fixtures = [
        Fixture(id: "copula", input: "imnida", expected: "입니다"),
        Fixture(id: "terminal-ne", input: "halle", expected: "핥네"),
        Fixture(id: "english-particle", input: "testreul sihaenghanda", expected: "test를 시행한다"),
        Fixture(id: "sentence", input: "jeoneun haksaengimnida", expected: "저는 학생입니다"),
        Fixture(id: "protected-url", input: "https://example.com/test", expected: "https://example.com/test"),
    ]

    static func main() {
        let arguments = CommandLine.arguments
        guard arguments.count == 3, arguments[1] == "--resources" else {
            FileHandle.standardError.write(Data("Usage: engine-smoke-tests --resources <Hanglish.app/Contents/Resources>\n".utf8))
            exit(2)
        }
        let resourcesURL = URL(fileURLWithPath: arguments[2], isDirectory: true)
        let manifestURL = resourcesURL.appendingPathComponent("engine-manifest.json")
        guard let manifestData = try? Data(contentsOf: manifestURL),
              let manifest = try? JSONSerialization.jsonObject(with: manifestData) as? [String: Any],
              let expectedVersion = manifest["engine_version"] as? String,
              !expectedVersion.isEmpty else {
            FileHandle.standardError.write(Data("Missing packaged engine identity manifest\n".utf8))
            exit(1)
        }
        let client = EngineClient(resourcesURL: resourcesURL)
        var failures: [String] = []
        var observedVersion: String?

        for (index, fixture) in fixtures.enumerated() {
            var state = CompositionState()
            guard state.append(fixture.input), let conversion = state.beginConversion() else {
                failures.append("\(fixture.id): could not begin composition")
                continue
            }

            let requestID = "macos-engine-smoke-\(index)"
            guard let result = waitForReply(
                client: client,
                input: conversion.input,
                requestID: requestID
            ) else {
                failures.append("\(fixture.id): timed out waiting for the local engine")
                continue
            }

            switch result {
            case .failure(let error):
                failures.append("\(fixture.id): \(error.localizedDescription)")
            case .success(let reply):
                if reply.request_id != requestID {
                    failures.append("\(fixture.id): response request ID did not match")
                }
                if reply.version.isEmpty {
                    failures.append("\(fixture.id): response version was empty")
                } else if reply.version != expectedVersion {
                    failures.append("\(fixture.id): response version did not match the packaged manifest")
                } else if let observedVersion, reply.version != observedVersion {
                    failures.append("\(fixture.id): response version changed during the smoke test")
                } else {
                    observedVersion = reply.version
                }

                guard state.accept(
                    candidates: reply.candidates.map(\.text),
                    revision: conversion.revision
                ) else {
                    failures.append("\(fixture.id): composition rejected a current response")
                    continue
                }
                if state.displayText != fixture.expected {
                    failures.append("\(fixture.id): first candidate did not match the expected result")
                }
                if state.finish() != fixture.expected {
                    failures.append("\(fixture.id): committed result did not match the preview")
                }
                if !state.isEmpty {
                    failures.append("\(fixture.id): finish did not clear composition state")
                }
            }
        }

        if failures.isEmpty {
            print("EngineSmokeTests: all \(fixtures.count) live conversions passed")
            print("EngineSmokeTests: one stable, nonempty engine version observed")
        } else {
            let report = failures.map { "FAIL: \($0)" }.joined(separator: "\n") + "\n"
            FileHandle.standardError.write(Data(report.utf8))
            exit(1)
        }
    }

    private static func waitForReply(
        client: EngineClient,
        input: String,
        requestID: String
    ) -> Result<ConversionReply, Error>? {
        var reply: Result<ConversionReply, Error>?
        let task = client.convert(input, requestID: requestID) { result in
            reply = result
        }
        let deadline = Date(timeIntervalSinceNow: 75)
        while reply == nil, Date() < deadline {
            _ = RunLoop.main.run(mode: .default, before: Date(timeIntervalSinceNow: 0.05))
        }
        if reply == nil {
            task.cancel()
        }
        return reply
    }
}
