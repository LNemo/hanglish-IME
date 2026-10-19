#!/bin/bash
set -euo pipefail
task_dir="$(cd "$(dirname "$0")" && pwd)"
repo_dir="$(cd "$task_dir/../.." && pwd)"
python="$repo_dir/.venv/bin/python"
[[ "$(uname -m)" == arm64 ]] || { echo 'This package targets Apple Silicon.' >&2; exit 1; }
[[ -x "$python" ]] || { echo 'Prepare .venv first; see native/macos/README.md.' >&2; exit 1; }
"$python" "$repo_dir/scripts/prepare_runtime_assets.py" check --profile package || {
  echo 'Runtime assets are missing or differ from the lock. See native/macos/README.md.' >&2
  exit 1
}
bash "$task_dir/build.sh"
mkdir -p "$task_dir/build" "$repo_dir/dist"
stage="$(mktemp -d "$task_dir/build/package.XXXXXX")"
installer="$stage/Hanglish Installer.app"
payload="$installer/Contents/Resources/Hanglish.app"
mkdir -p "$installer/Contents/MacOS" "$installer/Contents/Resources"
ditto "$task_dir/build/Hanglish.app" "$payload"
"$python" "$task_dir/Packaging/bundle_engine.py" --resources "$payload/Contents/Resources"
"$python" "$task_dir/Packaging/sign_payload.py" "$payload"
bash "$task_dir/test.sh" --live "$payload/Contents/Resources"
codesign --verify --deep --strict "$payload"
cp "$task_dir/Installer/Info.plist" "$installer/Contents/Info.plist"
/usr/libexec/PlistBuddy -c 'Set :LSMinimumSystemVersion 14.0' "$installer/Contents/Info.plist"
/usr/libexec/PlistBuddy -c 'Add :CFBundleIconFile string Hanglish.icns' "$installer/Contents/Info.plist"
cp "$payload/Contents/Resources/Hanglish.icns" "$installer/Contents/Resources/Hanglish.icns"
swiftc -swift-version 5 -O -target arm64-apple-macosx14.0 \
  -module-cache-path "$task_dir/build/module-cache" -framework Cocoa -framework Carbon \
  "$task_dir/Installer/Installer.swift" -o "$installer/Contents/MacOS/Hanglish Installer"
plutil -lint "$installer/Contents/Info.plist"
codesign --force --sign - "$installer"
codesign --verify --deep --strict "$installer"
cp "$task_dir/Packaging/THIRD_PARTY_NOTICES.md" "$stage/THIRD_PARTY_NOTICES.md"
cp "$repo_dir/LICENSE" "$stage/LICENSE"
cp "$repo_dir/NOTICE" "$stage/NOTICE"
cp "$task_dir/README.ko.md" "$stage/설치 안내.md"
version="$(/usr/libexec/PlistBuddy -c 'Print :CFBundleShortVersionString' "$payload/Contents/Info.plist")"
dmg="$repo_dir/dist/Hanglish-$version-arm64.dmg"
hdiutil create -ov -format UDZO -volname "Hanglish $version" -srcfolder "$stage" "$dmg"
hdiutil verify "$dmg"
shasum -a 256 "$dmg" > "$dmg.sha256"
printf 'Package: %s\nInstaller: %s\n' "$dmg" "$installer"
