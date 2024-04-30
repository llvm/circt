#!/bin/bash


for filename in ../massive_reachability_rtl/subset_for_testing/*; do
    [ -e "$filename" ] || continue
    echo $filename
	echo $filename >> times_10000.txt
	echo "Bound 10000"
    	for i in $(seq 1 10); do
    		rm temptime
    		/usr/bin/time -o temptime -f "%e" ../../../bea-circt/circt/build/bin/circt-mc -b 10000 $filename
		cat temptime >> times$j.txt
    done
done


