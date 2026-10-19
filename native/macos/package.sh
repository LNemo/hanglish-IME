#!/bin/bash
set -euo pipefail

task_dir="$(cd "$(dirname "$0")" && pwd)"
repo_dir="$(cd "$task_dir/../.." && pwd)"
python="$repo_dir/.venv/bin/python"
package_mode="${HANGLISH_PACKAGE_MODE:-development}"
signing_identity="${HANGLISH_SIGNING_IDENTITY:-}"
team_id="${HANGLISH_TEAM_ID:-}"
notary_profile="${HANGLISH_NOTARY_PROFILE:-}"
arch="$(uname -m)"
release_commit=""
cleanup_paths=()

cleanup() {
  local path
  for path in "${cleanup_paths[@]}"; do
    [[ -z "$path" ]] || rm -rf -- "$path"
  done
}
trap cleanup EXIT

case "$package_mode" in
  development|release) ;;
  *) echo 'HANGLISH_PACKAGE_MODE must be development or release.' >&2; exit 2 ;;
esac
[[ "$arch" == arm64 ]] || { echo 'This package targets Apple Silicon.' >&2; exit 1; }
[[ -x "$python" ]] || { echo 'Prepare .venv first; see native/macos/README.md.' >&2; exit 1; }
mkdir -p "$task_dir/build" "$repo_dir/dist"

version="$(/usr/libexec/PlistBuddy -c 'Print :CFBundleShortVersionString' "$task_dir/Info.plist")"
build_number="$(/usr/libexec/PlistBuddy -c 'Print :CFBundleVersion' "$task_dir/Info.plist")"
if [[ "$package_mode" == release ]]; then
  [[ -n "$signing_identity" && -n "$team_id" && -n "$notary_profile" ]] || {
    echo 'Release mode requires HANGLISH_SIGNING_IDENTITY, HANGLISH_TEAM_ID, and HANGLISH_NOTARY_PROFILE.' >&2
    exit 1
  }
  preflight_commit_file="$(mktemp "${TMPDIR:-/tmp}/hanglish-preflight.XXXXXX")"
  cleanup_paths+=("$preflight_commit_file")
  "$python" "$task_dir/Packaging/release_tools.py" preflight \
    --repo "$repo_dir" --identity "$signing_identity" --team-id "$team_id" \
    --profile "$notary_profile" --commit-output "$preflight_commit_file"
  IFS= read -r release_commit < "$preflight_commit_file"
  [[ -n "$release_commit" ]] || { echo 'Release preflight returned no source commit.' >&2; exit 1; }
fi

suffix=""
[[ "$package_mode" == development ]] && suffix="-development-NONPUBLIC"
dmg="$repo_dir/dist/Hanglish-$version-$arch$suffix.dmg"
checksum="$dmg.sha256"
manifest="$dmg.json"
notary_log="$dmg.notary-log.json"
for output in "$dmg" "$checksum" "$manifest"; do
  [[ ! -e "$output" ]] || { echo "Refusing to overwrite existing artifact: $output" >&2; exit 1; }
done
[[ "$package_mode" != release || ! -e "$notary_log" ]] || {
  echo "Refusing to overwrite existing artifact: $notary_log" >&2; exit 1
}

if [[ "$package_mode" == release ]]; then
  command -v uv >/dev/null || { echo 'Release packaging requires uv.' >&2; exit 1; }
  release_environment="$(mktemp -d "$task_dir/build/release-venv.XXXXXX")"
  cleanup_paths+=("$release_environment")
  env UV_PROJECT_ENVIRONMENT="$release_environment" \
    uv sync --project "$repo_dir" --frozen --extra learning --no-dev --python 3.11
  python="$release_environment/bin/python"
fi

"$python" "$repo_dir/scripts/prepare_runtime_assets.py" check --profile package || {
  echo 'Runtime assets are missing or differ from the lock. See native/macos/README.md.' >&2
  exit 1
}
bash "$task_dir/build.sh" --signing-mode unsigned

stage="$(mktemp -d "$task_dir/build/package.XXXXXX")"
artifact_stage="$(mktemp -d "$task_dir/build/artifact.XXXXXX")"
cleanup_paths+=("$stage" "$artifact_stage")
working_dmg="$artifact_stage/$(basename "$dmg")"
working_checksum="$artifact_stage/$(basename "$checksum")"
working_manifest="$artifact_stage/$(basename "$manifest")"
working_notary_log="$artifact_stage/$(basename "$notary_log")"
installer="$stage/Hanglish Installer.app"
payload="$installer/Contents/Resources/Hanglish.app"
mkdir -p "$installer/Contents/MacOS" "$installer/Contents/Resources"
ditto "$task_dir/build/Hanglish.app" "$payload"
"$python" "$task_dir/Packaging/bundle_engine.py" --resources "$payload/Contents/Resources"
if [[ "$package_mode" == release ]]; then
  "$python" "$task_dir/Packaging/sign_payload.py" \
    --release --identity "$signing_identity" "$payload"
else
  "$python" "$task_dir/Packaging/sign_payload.py" "$payload"
fi
bash "$task_dir/test.sh" --live "$payload/Contents/Resources"
codesign --verify --deep --strict --verbose=2 "$payload"

cp "$task_dir/Installer/Info.plist" "$installer/Contents/Info.plist"
/usr/libexec/PlistBuddy -c 'Set :LSMinimumSystemVersion 14.0' "$installer/Contents/Info.plist"
/usr/libexec/PlistBuddy -c "Set :CFBundleShortVersionString $version" "$installer/Contents/Info.plist"
/usr/libexec/PlistBuddy -c "Set :CFBundleVersion $build_number" "$installer/Contents/Info.plist"
/usr/libexec/PlistBuddy -c 'Add :CFBundleIconFile string Hanglish.icns' "$installer/Contents/Info.plist"
cp "$payload/Contents/Resources/Hanglish.icns" "$installer/Contents/Resources/Hanglish.icns"
[[ ! -d "$task_dir/Installer/Resources" ]] || \
  ditto "$task_dir/Installer/Resources" "$installer/Contents/Resources"
installer_sources=()
[[ -f "$task_dir/Shared/Localization.swift" ]] && installer_sources+=("$task_dir/Shared/Localization.swift")
installer_sources+=("$task_dir/Installer/Installer.swift")
swiftc -swift-version 5 -O -target arm64-apple-macosx14.0 \
  -module-cache-path "$task_dir/build/module-cache" -framework Cocoa -framework Carbon \
  "${installer_sources[@]}" -o "$installer/Contents/MacOS/Hanglish Installer"
plutil -lint "$installer/Contents/Info.plist"
if [[ "$package_mode" == release ]]; then
  codesign --force --sign "$signing_identity" --options runtime --timestamp "$installer"
else
  codesign --force --sign - "$installer"
fi
codesign --verify --deep --strict --verbose=2 "$installer"

cp "$task_dir/Packaging/THIRD_PARTY_NOTICES.md" "$stage/THIRD_PARTY_NOTICES.md"
cp "$repo_dir/LICENSE" "$stage/LICENSE"
cp "$repo_dir/NOTICE" "$stage/NOTICE"
cp "$task_dir/README.md" "$stage/Installation Guide.md"
cp "$task_dir/README.ko.md" "$stage/설치 안내.md"
if [[ "$package_mode" == development ]]; then
  printf '%s\n' \
    'NONPUBLIC DEVELOPMENT BUILD' \
    'This artifact uses ad-hoc signing and must not be published as a general release.' \
    > "$stage/NONPUBLIC-DEVELOPMENT-BUILD.txt"
fi
"$python" "$task_dir/Packaging/release_tools.py" audit-stage \
  --stage "$stage" --forbid-root "$repo_dir"
if [[ "$package_mode" == release ]]; then
  "$python" "$task_dir/Packaging/release_tools.py" verify-source \
    --repo "$repo_dir" --commit "$release_commit"
fi

hdiutil create -format UDZO -volname "Hanglish $version" -srcfolder "$stage" "$working_dmg"
if [[ "$package_mode" == release ]]; then
  codesign --force --sign "$signing_identity" --timestamp "$working_dmg"
  codesign --verify --verbose=2 "$working_dmg"
  "$python" "$task_dir/Packaging/release_tools.py" notarize \
    --dmg "$working_dmg" --profile "$notary_profile" --log "$working_notary_log"
  xcrun stapler staple "$working_dmg"
  "$python" "$task_dir/Packaging/release_tools.py" verify-release --dmg "$working_dmg"
else
  hdiutil verify "$working_dmg"
fi

digest="$(shasum -a 256 "$working_dmg" | awk '{print $1}')"
printf '%s  %s\n' "$digest" "$(basename "$dmg")" > "$working_checksum"
manifest_command=("$python" "$task_dir/Packaging/release_tools.py" manifest \
  --repo "$repo_dir" --plist "$task_dir/Info.plist" --artifact "$working_dmg" \
  --output "$working_manifest" --mode "$package_mode" --arch "$arch" \
  --input "$task_dir/Info.plist" \
  --input "$task_dir/Packaging/runtime-assets.lock.json" \
  --input "$task_dir/Packaging/engine-files.json" \
  --input "$task_dir/Packaging/THIRD_PARTY_NOTICES.md" \
  --input "$repo_dir/LICENSE" --input "$repo_dir/NOTICE" --input "$repo_dir/uv.lock" \
  --tree-input "bundled-python=$payload/Contents/Resources/Python" \
  --tree-input "bundled-engine=$payload/Contents/Resources/Engine")
if [[ "$package_mode" == release ]]; then
  manifest_command+=(--require-commit "$release_commit" --require-clean)
fi
"${manifest_command[@]}"
mv "$working_dmg" "$dmg"
mv "$working_checksum" "$checksum"
mv "$working_manifest" "$manifest"
[[ "$package_mode" != release ]] || mv "$working_notary_log" "$notary_log"
printf 'Package: %s\nChecksum: %s\nManifest: %s\n' "$dmg" "$checksum" "$manifest"
