#!/bin/bash
set -euo pipefail
task_dir="$(cd "$(dirname "$0")/.." && pwd)"
repo_dir="$(cd "$task_dir/../.." && pwd)"
[[ $# == 1 ]] || { echo 'Usage: test_portable.sh <bundled Resources directory>' >&2; exit 2; }
stage="$(mktemp -d /private/tmp/hanglish-portable.XXXXXX)"
trap 'rm -rf "$stage"' EXIT
resources="$stage/Relocated App/Contents/Resources"
ditto "$1" "$resources"
swiftc -swift-version 5 -module-cache-path "$task_dir/build/module-cache" \
  "$task_dir/Sources/CompositionState.swift" "$task_dir/Sources/EngineClient.swift" \
  "$task_dir/Tests/EngineSmokeTests.swift" -o "$stage/engine-smoke"
cat > "$stage/isolation.sb" <<'PROFILE'
(version 1)
(allow default)
(deny network*)
(deny file-read* (subpath (param "REPO")))
(deny file-read* (subpath (param "DEV_PYTHON")))
(deny file-read* (subpath (param "HF_CACHE")))
PROFILE
profile=(/usr/bin/sandbox-exec -D "REPO=$repo_dir" \
  -D "DEV_PYTHON=$HOME/.local/share/uv/python" -D "HF_CACHE=$HOME/.cache/huggingface" \
  -f "$stage/isolation.sb")
if "${profile[@]}" /bin/cat "$task_dir/Info.plist" >/dev/null 2>&1; then
  echo 'Isolation failed: repository remains readable' >&2; exit 1
fi
"${profile[@]}" "$resources/Python/bin/python3.11" -I -B -c '
import socket
try:
    socket.socket().connect(("127.0.0.1", 8765))
except PermissionError:
    print("Isolation: repository and network access denied")
else:
    raise AssertionError("Network denial was not enforced")
'
cd "$stage"
"${profile[@]}" "$stage/engine-smoke" --resources "$resources"
echo 'Relocated standalone engine: passed without repository, developer Python, model cache or network'
