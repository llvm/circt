#!/bin/bash


for j in $(seq 100 100 500); do
    rm times$j.txt
done

for filename in *_lowered.mlir; do
    [ -e "$filename" ] || continue
    echo $filename
    for j in $(seq 100 100 500); do
    	echo $filename >> times$j.txt
	echo "Bound $j"
    	for i in $(seq 1 10); do
    		rm temptime
    		/usr/bin/time -o temptime -f "%e" ../../circt-fork/build/bin/circt-mc -b $j $filename
		cat temptime >> times$j.txt
	done	
    done
done


