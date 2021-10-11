#!/usr/bin/bash

CELL=${1}
MODEL=${2}
for TF in $(ls ./$MODEL/)
do
    echo "working on $TF."
    if [ ! -d ./fasta/$CELL/$TF ]; then
        mkdir -p ./fasta/$CELL/$TF
    fi
    
    python TFBS_locating.py -d `pwd`/$CELL/$TF/data \
                             -n $TF \
                             -g 0 \
                             -t 0.2 \
                             -c `pwd`/$MODEL/$TF \
                             -o `pwd`/fasta/$CELL/$TF
done
