#!/bin/bash
set -euo pipefail

task_dir="$(cd "$(dirname "$0")" && pwd)"
build_dir="$task_dir/build"
app_dir="$build_dir/Hanglish.app"
signing_mode=development
signing_identity=-

while [[ $# -gt 0 ]]; do
  case "$1" in
    --signing-mode) signing_mode="${2:-}"; shift 2 ;;
    --identity) signing_identity="${2:-}"; shift 2 ;;
    *)
      printf 'Usage: %s [--signing-mode unsigned|development|release] [--identity NAME]\n' "$0" >&2
      exit 2
      ;;
  esac
done

case "$signing_mode" in
  unsigned|development) ;;
  release)
    [[ "$signing_identity" == 'Developer ID Application:'* ]] || {
      echo 'Release builds require --identity with a Developer ID Application identity.' >&2
      exit 1
    }
    ;;
  *) echo "Unknown signing mode: $signing_mode" >&2; exit 2 ;;
esac

mkdir -p "$build_dir" "$build_dir/module-cache"
stage="$(mktemp -d "$build_dir/app.XXXXXX")"
trap 'rm -rf "$stage"' EXIT
stage_app="$stage/Hanglish.app"
mkdir -p "$stage_app/Contents/MacOS" "$stage_app/Contents/Resources"
cp "$task_dir/Info.plist" "$stage_app/Contents/Info.plist"
ditto "$task_dir/Resources" "$stage_app/Contents/Resources"
rm -f "$stage_app/Contents/Resources/HanglishGlyph.pdf"

sources=("$task_dir/Sources/CompositionState.swift" "$task_dir/Sources/EngineClient.swift")
[[ -f "$task_dir/Shared/Localization.swift" ]] && sources+=("$task_dir/Shared/Localization.swift")
sources+=("$task_dir/Sources/InputController.swift" "$task_dir/Sources/main.swift")
swiftc -swift-version 5 -module-cache-path "$build_dir/module-cache" \
  -target "$(uname -m)-apple-macosx14.0" -O \
  -framework Cocoa -framework InputMethodKit \
  "${sources[@]}" -o "$stage_app/Contents/MacOS/Hanglish"
swiftc -swift-version 5 -module-cache-path "$build_dir/module-cache" \
  "$task_dir/Tools/MakeIcon.swift" -o "$build_dir/make-icon"
"$build_dir/make-icon" "$stage_app/Contents/Resources/Hanglish.tiff"
plutil -lint "$stage_app/Contents/Info.plist"

case "$signing_mode" in
  development) python3 "$task_dir/Packaging/sign_payload.py" "$stage_app" ;;
  release)
    python3 "$task_dir/Packaging/sign_payload.py" \
      --release --identity "$signing_identity" "$stage_app"
    ;;
esac

rm -rf "$app_dir"
ditto "$stage_app" "$app_dir"
swiftc -swift-version 5 -module-cache-path "$build_dir/module-cache" -framework Carbon \
  "$task_dir/Tools/InputSourceTool.swift" -o "$build_dir/input-source"
printf 'Built %s (%s signing)\n' "$app_dir" "$signing_mode"
