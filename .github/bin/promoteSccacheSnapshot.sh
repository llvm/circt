#!/usr/bin/env bash

set -euo pipefail

if (( $# != 1 )); then
  echo "usage: $0 <artifact-directory>" >&2
  exit 2
fi

readonly artifact_dir=$1
readonly cache=/var/cache/circt-sccache
readonly publish=/var/cache/circt-sccache-publish
readonly next=/var/cache/circt-sccache-next
readonly previous=/var/cache/circt-sccache-previous

[[ "$artifact_dir" == /* ]]
test -d "$artifact_dir"

as_root() {
  if (( EUID == 0 )); then
    "$@"
  else
    sudo "$@"
  fi
}

# A runner interrupted between the two renames may leave the old cache under
# the rollback name. Restore it before validating either snapshot directory.
if [[ ! -d "$cache" && -d "$previous" ]]; then
  as_root mv "$previous" "$cache"
fi

as_root chmod -R a+rwX "$cache" "$publish"
test -f "$publish/.ready"
find "$publish" -type f ! -name .ready -print -quit | grep -q .
as_root find "$publish" -maxdepth 1 -name .ready -delete

for stale in "$next" "$previous"; do
  if [[ -d "$stale" ]]; then
    as_root find "$stale" -mindepth 1 -delete
    as_root rmdir "$stale"
  fi
done

as_root install -d -m 0777 "$next"
as_root cp -R --no-preserve=ownership,mode,timestamps "$publish/." "$next/"
find "$next" -type f -print -quit | grep -q .
as_root chmod -R a+rwX "$next"

recover_cache() {
  if [[ ! -d "$cache" && -d "$previous" ]]; then
    as_root mv "$previous" "$cache"
  fi
}
trap recover_cache EXIT
as_root mv "$cache" "$previous"
as_root mv "$next" "$cache"
trap - EXIT

as_root find "$previous" -mindepth 1 -delete
as_root rmdir "$previous"
as_root find "$publish" -mindepth 1 -delete
find "$cache" -type f -print -quit | grep -q .
du -sh "$cache" > "$artifact_dir/sccache-size-after-compile.txt"
