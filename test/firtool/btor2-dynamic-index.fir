; RUN: firtool --btor2 --split-input-file %s | FileCheck %s --check-prefixes=CHECK,OPT
; RUN: firtool --btor2 --split-input-file --disable-opt %s | FileCheck %s --check-prefixes=CHECK,NOOPT

FIRRTL version 4.0.0
circuit DynamicRead :
  public module DynamicRead :
    input clock : Clock
    input index : UInt<4>
    input values : UInt<8>[3]
    input expected : UInt<8>

    node selected = values[index]
    assert(clock, eq(selected, expected), UInt<1>(1), "")

; CHECK-LABEL: START OF MODULE: DynamicRead
; CHECK:  1 sort bitvec 4
; CHECK:  2 input 1 index
; CHECK:  3 sort bitvec 8
; CHECK:  4 input 3 values_0
; CHECK:  5 input 3 values_1
; CHECK:  6 input 3 values_2
; CHECK:  7 input 3 expected
; CHECK:  8 sort bitvec 1
; Only the low two index bits are used. Index 3 selects element 0, matching
; the padding inserted by FIRRTL-to-HW lowering for non-power-of-two vectors.
; CHECK:  9 slice 8 2 0 0
; CHECK:  10 slice 8 2 1 1
; CHECK:  11 ite 3 9 4 6
; CHECK:  12 ite 3 9 5 4
; CHECK:  13 ite 3 10 11 12
; CHECK:  14 eq 8 13 7
; CHECK:  15 not 8 14
; CHECK:  16 bad 15

; // -----

FIRRTL version 4.0.0
circuit NestedRead :
  public module NestedRead :
    input clock : Clock
    input row : UInt<1>
    input column : UInt<2>
    input values : UInt<8>[4][2]
    input expected : UInt<8>

    node selected = values[row][column]
    assert(clock, eq(selected, expected), UInt<1>(1), "")

; CHECK-LABEL: START OF MODULE: NestedRead
; CHECK:  1 sort bitvec 1
; CHECK:  2 input 1 row
; CHECK:  3 sort bitvec 2
; CHECK:  4 input 3 column
; CHECK:  5 sort bitvec 8
; CHECK:  6 input 5 values_0_0
; CHECK:  7 input 5 values_0_1
; CHECK:  8 input 5 values_0_2
; CHECK:  9 input 5 values_0_3
; CHECK:  10 input 5 values_1_0
; CHECK:  11 input 5 values_1_1
; CHECK:  12 input 5 values_1_2
; CHECK:  13 input 5 values_1_3
; CHECK:  14 input 5 expected
; Optimized FIRRTL selects a whole row before selecting a column.
; OPT:  15 sort bitvec 32
; OPT:  16 sort bitvec 16
; OPT:  17 concat 16 13 12
; OPT:  18 sort bitvec 24
; OPT:  19 concat 18 17 11
; OPT:  20 concat 15 19 10
; OPT:  21 concat 16 9 8
; OPT:  22 concat 18 21 7
; OPT:  23 concat 15 22 6
; OPT:  24 ite 15 2 20 23
; OPT:  25 slice 5 24 7 0
; OPT:  26 slice 5 24 15 8
; OPT:  27 slice 5 24 23 16
; OPT:  28 slice 5 24 31 24
; OPT:  29 slice 1 4 0 0
; OPT:  30 slice 1 4 1 1
; OPT:  31 ite 5 29 28 27
; OPT:  32 ite 5 29 26 25
; OPT:  33 ite 5 30 31 32
; OPT:  34 eq 1 33 14
; OPT:  35 not 1 34
; OPT:  36 bad 35
; Without FIRRTL optimization, each column selects its row independently.
; NOOPT:  15 ite 5 2 10 6
; NOOPT:  16 ite 5 2 11 7
; NOOPT:  17 ite 5 2 12 8
; NOOPT:  18 ite 5 2 13 9
; NOOPT:  19 slice 1 4 0 0
; NOOPT:  20 slice 1 4 1 1
; NOOPT:  21 ite 5 19 18 17
; NOOPT:  22 ite 5 19 16 15
; NOOPT:  23 ite 5 20 21 22
; NOOPT:  24 eq 1 23 14
; NOOPT:  25 not 1 24
; NOOPT:  26 bad 25

; // -----

FIRRTL version 4.0.0
circuit DynamicWrite :
  public module DynamicWrite :
    input clock : Clock
    input index : UInt<4>
    input values : UInt<8>[3]
    input value : UInt<8>
    input expected : UInt<8>

    wire updated : UInt<8>[3]
    connect updated, values
    connect updated[index], value
    assert(clock, eq(updated[1], expected), UInt<1>(1), "")

; CHECK-LABEL: START OF MODULE: DynamicWrite
; CHECK:  1 sort bitvec 4
; CHECK:  2 input 1 index
; CHECK:  3 sort bitvec 8
; CHECK:  4 input 3 values_0
; CHECK:  5 input 3 values_1
; CHECK:  6 input 3 values_2
; CHECK:  7 input 3 value
; CHECK:  8 input 3 expected
; Compare the full-width write index: out-of-range writes must have no effect.
; CHECK:  9 constd 1 1
; CHECK:  10 sort bitvec 1
; CHECK:  11 eq 10 2 9
; CHECK:  12 ite 3 11 7 5
; CHECK:  13 eq 10 12 8
; CHECK:  14 not 10 13
; CHECK:  15 bad 14

; // -----

FIRRTL version 4.0.0
circuit SmallReads :
  public module SmallReads :
    input clock : Clock
    input index : UInt<1>
    input wideIndex : UInt<4>
    input single : UInt<8>[1]
    input pair : UInt<8>[2]
    input values : UInt<8>[4]
    input expected : UInt<8>

    assert(clock, eq(single[wideIndex], expected), UInt<1>(1), "")
    assert(clock, eq(pair[wideIndex], expected), UInt<1>(1), "")
    assert(clock, eq(values[index], expected), UInt<1>(1), "")

; CHECK-LABEL: START OF MODULE: SmallReads
; CHECK:  1 sort bitvec 1
; CHECK:  2 input 1 index
; CHECK:  3 sort bitvec 4
; CHECK:  4 input 3 wideIndex
; CHECK:  5 sort bitvec 8
; CHECK:  6 input 5 single_0
; CHECK:  7 input 5 pair_0
; CHECK:  8 input 5 pair_1
; CHECK:  9 input 5 values_0
; CHECK:  10 input 5 values_1
; CHECK:  11 input 5 values_2
; CHECK:  12 input 5 values_3
; CHECK:  13 input 5 expected
; CHECK:  14 eq 1 6 13
; CHECK:  15 slice 1 4 0 0
; CHECK:  16 ite 5 15 8 7
; CHECK:  17 eq 1 16 13
; A narrow index can only select the first two elements of the larger vector.
; CHECK:  18 ite 5 2 10 9
; CHECK:  19 eq 1 18 13
; CHECK:  20 not 1 14
; CHECK:  21 bad 20
; CHECK:  22 not 1 17
; CHECK:  23 bad 22
; CHECK:  24 not 1 19
; CHECK:  25 bad 24
