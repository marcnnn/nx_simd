#!/bin/sh
# Clones the sibling libraries next to this one (path dependencies).
set -e
BASE=${SIBLINGS_BASE:-https://github.com/$(dirname "${GITHUB_REPOSITORY:-x/y}")}
[ -d ../nx_fixed ] || git clone --depth 1 "$BASE/nx_fixed.git" ../nx_fixed
