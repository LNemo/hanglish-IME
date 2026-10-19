#!/bin/bash
set -euo pipefail

task_dir="$(cd "$(dirname "$0")/.." && pwd)"
fixture_app="$task_dir/build/Hanglish.app"

if [[ ! -x "$fixture_app/Contents/MacOS/Hanglish" ]]; then
  bash "$task_dir/build.sh"
fi

test_root="$(mktemp -d /tmp/hanglish-installer-test.XXXXXX)"
trap 'rm -rf "$test_root"' EXIT

installer_app="$test_root/Hanglish Installer.app"
installer_binary="$installer_app/Contents/MacOS/Hanglish Installer"
payload="$installer_app/Contents/Resources/Hanglish.app"
destination="$test_root/destination/Hanglish.app"
backup_root="$test_root/backups"
module_cache="$test_root/module-cache"

mkdir -p "$installer_app/Contents/MacOS" "$installer_app/Contents/Resources" "$module_cache"
cp "$task_dir/Installer/Info.plist" "$installer_app/Contents/Info.plist"
ditto "$task_dir/Installer/Resources" "$installer_app/Contents/Resources"
swiftc -swift-version 5 -module-cache-path "$module_cache" \
  -target "$(uname -m)-apple-macosx13.0" \
  -framework Cocoa -framework Carbon \
  "$task_dir/Shared/Localization.swift" \
  "$task_dir/Installer/Installer.swift" \
  -o "$installer_binary"

# The executable must select its bundled Korean resources when macOS prefers Korean.
if "$installer_binary" -AppleLanguages '(ko)' --inspect --destination \
    >"$test_root/korean.out" 2>"$test_root/korean.err"; then
  printf 'Expected a missing destination argument failure.\n' >&2
  exit 1
fi
grep -q '뒤에 경로가 필요합니다' "$test_root/korean.err"

make_payload() {
  local version="$1"
  rm -rf "$payload"
  cp -R "$fixture_app" "$payload"
  printf '%s\n' "$version" > "$payload/Contents/Resources/installer-test-version.txt"
  codesign --force --deep --sign - "$payload" >/dev/null
  codesign --verify --deep --strict "$payload"
}

run_installer() {
  "$installer_binary" -AppleLanguages '(en)' "$@" \
    --destination "$destination" \
    --backup-root "$backup_root" \
    --no-register
}

make_payload "version-1"
run_installer --install >/dev/null
test "$(tr -d '\n' < "$destination/Contents/Resources/installer-test-version.txt")" = "version-1"
run_installer --inspect | /usr/bin/python3 -c \
  'import json,sys; value=json.load(sys.stdin); assert value["installed"] is True; assert value["bundleIdentifier"] == "org.hanglish.inputmethod.Hanglish"; assert value["inputSourceIdentifiers"] == ["org.hanglish.inputmethod.Hanglish.Han", "org.hanglish.inputmethod.Hanglish"]; assert value["inputSourceEnablementOrder"] == ["org.hanglish.inputmethod.Hanglish", "org.hanglish.inputmethod.Hanglish.Han"]'

make_payload "version-2"
run_installer --install >/dev/null
test "$(tr -d '\n' < "$destination/Contents/Resources/installer-test-version.txt")" = "version-2"
old_backup="$(find "$backup_root" -path '*/Hanglish.app/Contents/Resources/installer-test-version.txt' -print -quit)"
test -n "$old_backup"
test "$(tr -d '\n' < "$old_backup")" = "version-1"

# A damaged signature must fail before the working installation is moved.
printf '%s\n' 'tampered-after-signing' > "$payload/Contents/Resources/installer-test-version.txt"
if run_installer --install >"$test_root/signature.out" 2>"$test_root/signature.err"; then
  printf 'Expected signature validation failure.\n' >&2
  exit 1
fi
test "$(tr -d '\n' < "$destination/Contents/Resources/installer-test-version.txt")" = "version-2"
grep -q 'code signature' "$test_root/signature.err"

# An unrelated app at the destination must never be overwritten.
unsafe_destination="$test_root/unsafe/Hanglish.app"
mkdir -p "$test_root/unsafe"
cp -R "$installer_app" "$unsafe_destination"
make_payload "version-3"
if "$installer_binary" -AppleLanguages '(en)' --install \
    --destination "$unsafe_destination" \
    --backup-root "$backup_root" \
    --no-register >"$test_root/unrelated.out" 2>"$test_root/unrelated.err"; then
  printf 'Expected unrelated bundle protection failure.\n' >&2
  exit 1
fi
test "$(/usr/libexec/PlistBuddy -c 'Print :CFBundleIdentifier' "$unsafe_destination/Contents/Info.plist")" = "org.hanglish.inputmethod.Installer"
grep -q 'not a Hanglish app' "$test_root/unrelated.err"

# Removal is recoverable: it moves the installation into a new backup.
run_installer --uninstall >/dev/null
test ! -e "$destination"
removed_backup="$(find "$backup_root" -path '*/Hanglish.app/Contents/Resources/installer-test-version.txt' -exec grep -l '^version-2$' {} \; -quit)"
test -n "$removed_backup"

# The package remains usable after removal and can install a fresh copy.
run_installer --install >/dev/null
test "$(tr -d '\n' < "$destination/Contents/Resources/installer-test-version.txt")" = "version-3"

printf 'Hanglish installer lifecycle tests: passed\n'
