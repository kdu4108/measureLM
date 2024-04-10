#!/bin/bash
# cd ../..; # the python scripts are all in the root directory.
echo ${which python};
echo $PWD;
echo "previewing args";
echo $1 -P $2 -S $3 -M $4 -Q $5 -MC $6 -ME $7 -ET $8 -QT $9 -CT ${10} -AM ${11} -ES ${12} -BS ${13} ${14} ${15} ${16} ${17} ${18} ${19} ${20}; # last 7 are boolean flags
python main.py $1 -P $2 -S $3 -M $4 -Q $5 -MC $6 -ME $7 -ET $8 -QT $9 -CT ${10} -AM ${11} -ES ${12} -BS ${13} ${14} ${15} ${16} ${17} ${18} ${19} ${20}; # last 7 are boolean flags
