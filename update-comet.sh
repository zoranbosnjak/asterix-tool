#!/usr/bin/env bash

# run all updaters
for i in $(find ast-tool* | grep update-comet.sh); do
    cd $(dirname $i)
    ./update-comet.sh
    cd ..
done;

