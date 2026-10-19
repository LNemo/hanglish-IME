#!/bin/bash
set -euo pipefail
task_dir="$(cd "$(dirname "$0")" && pwd)"
source_app="$task_dir/build/Hanglish.app"
destination="$HOME/Library/Input Methods/Hanglish.app"
test -x "$source_app/Contents/MacOS/Hanglish" || { printf 'Run bash native/macos/build.sh first.\n' >&2; exit 1; }
test -x "$source_app/Contents/Resources/Python/bin/python3.11" || {
  printf 'This lightweight build has no engine. Run package.sh and use the DMG installer.\n' >&2
  exit 2
}
codesign --verify --strict "$source_app"
mkdir -p "$(dirname "$destination")"
if [[ -e "$destination" ]]; then
  printf 'An installed Hanglish.app already exists. Move it aside before updating; see README.\n' >&2
  exit 1
fi
ditto "$source_app" "$destination"
"$task_dir/build/input-source" register "$destination"
if ! "$task_dir/build/input-source" enable org.hanglish.inputmethod.Hanglish; then
  "$task_dir/build/input-source" status
  printf 'App installed; activation is pending. See the Keyboard settings steps in native/macos/README.md.\n'
  exit 3
fi
"$task_dir/build/input-source" status
printf 'Installed. Select Hanglish from the macOS input menu to use it.\n'
