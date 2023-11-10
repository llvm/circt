// RUN: circt-opt %s --convert-hw-to-btor2 -o tmp.mlir | FileCheck %s  

//CHECK:    1 sort bitvec 1
//CHECK:    2 input 1 reset
//CHECK:    3 input 1 en
//CHECK:    4 sort bitvec 28
//CHECK:    5 constd 4 0
//CHECK:    6 constd 1 0
//CHECK:    7 sort bitvec 32
//CHECK:    8 constd 7 22
//CHECK:    9 constd 1 -1
//CHECK:    10 sort bitvec 4
//CHECK:    11 constd 10 -6
//CHECK:    12 constd 7 0
//CHECK:    13 state 7 count
//CHECK:    14 eq 1 13 8
//CHECK:    15 and 1 14 3
//CHECK:    16 ite 7 15 12 13
//CHECK:    17 neq 1 13 8
//CHECK:    18 and 1 17 3
//CHECK:    19 sort bitvec 33
//CHECK:    20 concat 19 6 13
//CHECK:    21 constd 19 1
//CHECK:    22 add 19 20 21
//CHECK:    23 slice 7 22 31 0
//CHECK:    24 ite 7 18 23 16
//CHECK:    25 constd 7 10
//CHECK:    26 neq 1 13 25
//CHECK:    27 implies 1 3 26
//CHECK:    28 not 1 27
//CHECK:    29 bad 28
//CHECK:    30 zero 7
//CHECK:    31 ite 7 2 30 24
//CHECK:    32 next 7 13 24
module {
    hw.module @Counter(in %clock : !seq.clock, in %reset : i1, in %en : i1) {
    %0 = seq.from_clock %clock
    %c0_i28 = hw.constant 0 : i28
    %false = hw.constant false
    %c22_i32 = hw.constant 22 : i32
    %true = hw.constant true
    %c-6_i4 = hw.constant -6 : i4
    %c0_i32 = hw.constant 0 : i32
    %count = seq.firreg %9 clock %clock reset sync %reset, %c0_i32 {firrtl.random_init_start = 0 : ui64} : i32
    %1 = comb.icmp bin eq %count, %c22_i32 : i32
    %2 = comb.and bin %1, %en : i1
    %3 = comb.mux bin %2, %c0_i32, %count : i32
    %4 = comb.icmp bin ne %count, %c22_i32 : i32
    %5 = comb.and bin %4, %en : i1
    %6 = comb.concat %false, %count : i1, i32
    %c1_i33 = hw.constant 1 : i33
    %7 = comb.add bin %6, %c1_i33 : i33
    %8 = comb.extract %7 from 0 : (i33) -> i32
    %9 = comb.mux bin %5, %8, %3 : i32
    %c10_i32 = hw.constant 10 : i32
    %10 = comb.icmp bin ne %count, %c10_i32 : i32
    sv.always posedge %0 {
      sv.if %en {
        sv.assert %10, immediate message "Counter reached 10!"
      }
    }
    hw.output
  }
}