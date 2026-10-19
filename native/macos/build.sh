#!/bin/bash
set -euo pipefail
task_dir="$(cd "$(dirname "$0")" && pwd)"
build_dir="$task_dir/build"
app_dir="$build_dir/Hanglish.app"
mkdir -p "$app_dir/Contents/MacOS" "$app_dir/Contents/Resources" "$build_dir/module-cache"
cp "$task_dir/Info.plist" "$app_dir/Contents/Info.plist"
ditto "$task_dir/Resources" "$app_dir/Contents/Resources"
# Remove obsolete generated icons when rebuilding an existing staging app.
rm -f "$app_dir/Contents/Resources/HanglishGlyph.pdf"
swiftc -swift-version 5 -module-cache-path "$build_dir/module-cache" \
  -target "$(uname -m)-apple-macosx14.0" -O \
  -framework Cocoa -framework InputMethodKit \
  "$task_dir/Sources/CompositionState.swift" "$task_dir/Sources/EngineClient.swift" \
  "$task_dir/Sources/InputController.swift" "$task_dir/Sources/main.swift" \
  -o "$app_dir/Contents/MacOS/Hanglish"
swiftc -swift-version 5 -module-cache-path "$build_dir/module-cache" \
  "$task_dir/Tools/MakeIcon.swift" -o "$build_dir/make-icon"
"$build_dir/make-icon" "$app_dir/Contents/Resources/Hanglish.tiff"
plutil -lint "$app_dir/Contents/Info.plist"
codesign --force --sign - "$app_dir"
codesign --verify --strict "$app_dir"
swiftc -swift-version 5 -module-cache-path "$build_dir/module-cache" -framework Carbon \
  "$task_dir/Tools/InputSourceTool.swift" -o "$build_dir/input-source"
printf 'Built %s\n' "$app_dir"
