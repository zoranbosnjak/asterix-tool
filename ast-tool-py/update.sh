#! /usr/bin/env nix-shell
#! nix-shell -i bash

set -e
export LC_ALL=C.UTF-8

changes1=$(git status --short .)
if [ -n "$changes1" ]; then
    echo "Error: local changes"
    exit 1
fi

# Bump version
changes2=$(git status --short ../nix/asterix-libs.json)
target=src/main.py
if [ -n "$changes2" ]; then
    echo "Bumping version"
    current_version=$(cat ${target} | grep "__version.*=" | sed 's/[^"]*//' | sed 's/\"//g')
    IFS='.' read -r -a array <<< "$current_version"
    new_version="${array[0]}.${array[1]}.$((array[2]+1))"
    sed -i -e "s/__version__ = \".*/__version__ = \"$new_version\"/g" ${target}
fi

