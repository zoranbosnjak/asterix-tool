#!/usr/bin/env bash

# exit when any command fails
set -e

src="https://github.com/zoranbosnjak/asterix-libs.git"
dst=./nix/asterix-libs.json

changes1=$(git status --short ${dst})
if [ -n "$changes1" ]; then
    echo "Error: local changes in ${dst}"
    exit 1
fi

nix-shell -p nix-prefetch-scripts --run "nix-prefetch-git ${src} > ${dst}"

changes2=$(git status --short ${dst})
if [ -n "$changes2" ]; then
    # run all updaters
    for i in $(find ast-tool* -type f | grep update.sh); do
        cd $(dirname $i)
        ./update.sh
        cd -
    done;
fi

