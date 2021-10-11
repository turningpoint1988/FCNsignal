#!/usr/bin/bash

Datadir=${1}
model=${2}
for experiment in $(ls ./$model/)
do
    echo "working on ${experiment}."
    python test_signal.py -d `pwd`/$Datadir/$experiment/data \
                          -n $experiment \
                          -g 0 \
                          -c `pwd`/$model/$experiment
done
