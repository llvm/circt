#!/usr/bin/env bash

#echo "Comparing $1 and $2 with $3"
yosys -q -p "
 read_verilog $1
 rename $3 top1
 proc
 memory
 flatten top1
 hierarchy -top top1
read_verilog $2
 rename $3 top2
 proc
 memory
 flatten top2
 equiv_make top1 top2 equiv
 hierarchy -top equiv
 clean -purge
 equiv_simple -short -v
 equiv_induct
 equiv_status -assert
"
if [ $? -eq 0 ]
then
  echo "PASS"
  exit 0
else
  echo "FAIL"
  exit 1
fi

