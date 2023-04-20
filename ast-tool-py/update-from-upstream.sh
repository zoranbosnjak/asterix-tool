#!/usr/bin/env bash

# update asterix lib
curl https://zoranbosnjak.github.io/asterix-lib-generator/lib/python/asterix.py --silent --output src/asterix.py

# increment version from x.y.z to x.(y+1).0
update=$(git status --short src/asterix.py)
if [ -n "$update" ]; then
    # extract current (x,y) from version string
    ver=$(cat pyproject.toml | grep "version.*=" | cut -f2- -d"=" | tr -d '"' | xargs)
    a=$(echo $ver | cut --delimiter="." -f 1)
    b=$(echo $ver | cut --delimiter="." -f 2)
    sed -i "s/^version.*/version = \"$a.$((b+1)).0\"/" pyproject.toml
fi

