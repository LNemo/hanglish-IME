#!/bin/bash
set -euo pipefail

task_dir="$(cd "$(dirname "$0")" && pwd)"
test_dir="$(mktemp -d "${TMPDIR:-/tmp}/hanglish-macos-tests.XXXXXX")"
trap 'rm -rf "$test_dir"' EXIT
module_cache="$test_dir/module-cache"
mkdir -p "$module_cache"

swift_flags=(
  -warnings-as-errors
  -swift-version 5
  -module-cache-path "$module_cache"
)

swiftc "${swift_flags[@]}" \
  "$task_dir/Sources/CompositionState.swift" \
  "$task_dir/Tests/CompositionStateTests.swift" \
  -o "$test_dir/composition-state-tests"
"$test_dir/composition-state-tests"

swiftc "${swift_flags[@]}" \
  "$task_dir/Sources/EngineClient.swift" \
  "$task_dir/Tests/EngineClientTests.swift" \
  -o "$test_dir/engine-client-tests"
"$test_dir/engine-client-tests"

swiftc "${swift_flags[@]}" \
  -target "$(uname -m)-apple-macosx14.0" \
  -framework Cocoa -framework InputMethodKit \
  -typecheck \
  "$task_dir/Sources/CompositionState.swift" \
  "$task_dir/Sources/EngineClient.swift" \
  "$task_dir/Sources/InputController.swift" \
  "$task_dir/Sources/main.swift"
printf 'Hanglish macOS app sources: typecheck passed\n'

case "${1:-}" in
  "") ;;
  --live)
    [[ $# -eq 2 ]] || { printf 'Usage: %s --live <resources-directory>\n' "$0" >&2; exit 2; }
    swiftc "${swift_flags[@]}" \
      "$task_dir/Sources/CompositionState.swift" \
      "$task_dir/Sources/EngineClient.swift" \
      "$task_dir/Tests/EngineSmokeTests.swift" \
      -o "$test_dir/engine-smoke-tests"
    "$test_dir/engine-smoke-tests" --resources "$2"
    ;;
  *)
    printf 'Usage: %s [--live <resources-directory>]\n' "$0" >&2
    exit 2
    ;;
esac
