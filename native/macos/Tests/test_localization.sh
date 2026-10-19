#!/bin/bash
set -euo pipefail

task_dir="$(cd "$(dirname "$0")/.." && pwd)"
test_root="$(mktemp -d /tmp/hanglish-localization-test.XXXXXX)"
trap 'rm -rf "$test_root"' EXIT

swiftc -swift-version 5 -module-cache-path "$test_root/module-cache" \
  "$task_dir/Shared/Localization.swift" \
  "$task_dir/Tests/LocalizationTests.swift" \
  -o "$test_root/localization-tests"

"$test_root/localization-tests" \
  "$task_dir/Installer/Resources" \
  "$task_dir/Resources"

plutil -lint "$task_dir/Installer/Info.plist" \
  "$task_dir/Installer/Resources/en.lproj/InfoPlist.strings" \
  "$task_dir/Installer/Resources/ko.lproj/InfoPlist.strings" \
  "$task_dir/Installer/Resources/en.lproj/Localizable.strings" \
  "$task_dir/Installer/Resources/ko.lproj/Localizable.strings" \
  "$task_dir/Resources/en.lproj/Localizable.strings" \
  "$task_dir/Resources/ko.lproj/Localizable.strings" >/dev/null

printf 'Hanglish localization resource validation: passed\n'
