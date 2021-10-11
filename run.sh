#!/usr/bin/bash

CELL=${1}
for TF in $(ls ./$CELL/)
do
    
    echo "working on $TF."
    if [ ! -d ./models/$TF ]; then
        mkdir -p ./models/$TF
    else
        continue
    fi
   
    python run_signal.py -d `pwd`/$CELL/$TF/data \
                         -n $TF \
                         -g 0 \
                         -b 500 \
                         -e 50 \
                         -c `pwd`/models/$TF
done


