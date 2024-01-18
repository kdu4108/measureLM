#!/bin/bash
# cd ../..; # the python scripts are all in the root directory.
echo $PWD;
python susceptibility_scores.py $1 -P $2 -S $3 -M $4 -Q $5 -MC $6 -ME $7 -ET $8 $9 ${10} ${11}; # last 3 are boolean flags
