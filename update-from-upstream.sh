#!/usr/bin/env bash

# run all updaters
for i in $(find ast-tool* | grep update-from-upstream.sh); do
    cd $(dirname $i)
    ./update-from-upstream.sh
    cd ..
done;

