#!/usr/bin/env bash
# Build the sandbox image from the current VeryDiff checkout.
#   agent/sandbox/build.sh [image-tag]
# Uses the host's HTTPS proxy / CA bundle (if set) during the build only.
set -euo pipefail
here="$(cd "$(dirname "$0")" && pwd)"
repo="$(cd "$here/../.." && pwd)"
tag="${1:-verydiff-agent-sandbox:latest}"
ctx="$(mktemp -d)"
trap 'rm -rf "$ctx"' EXIT
cp "$here/Dockerfile" "$ctx/"
mkdir -p "$ctx/verydiff"
cp -r "$repo/src" "$repo/Project.toml" "$repo/Manifest.toml" "$repo/cli.jl" "$ctx/verydiff/"
depot="${VDA_BUILD_DEPOT:-$HOME/.julia}"
mkdir -p "$ctx/depot"
for d in registries packages artifacts; do
  [ -d "$depot/$d" ] && tar -C "$depot" --exclude=.git -cf - "$d" | tar -C "$ctx/depot" -xf -
done
ca="${VDA_BUILD_CA:-/root/.ccr/ca-bundle.crt}"
[ -f "$ca" ] || ca=/etc/ssl/certs/ca-certificates.crt
args=(--network host --secret "id=ca,src=$ca" -t "$tag")
[ -n "${HTTPS_PROXY:-}" ] && args+=(--build-arg "HTTPS_PROXY=$HTTPS_PROXY" --build-arg "https_proxy=$HTTPS_PROXY")
DOCKER_BUILDKIT=1 docker build "${args[@]}" "$ctx"
echo "built $tag"
