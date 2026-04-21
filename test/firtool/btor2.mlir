// RUN: firtool --btor2 %s | FileCheck %s

hw.module @PastIntegration(in %clk: !seq.clock, in %rst: i1) {
  %clk_0 = seq.from_clock %clk
  %c0_i32 = hw.constant 0 : i32
  %c1_i32 = hw.constant 1 : i32
  %c5_i32 = hw.constant 5 : i32
  %init = seq.initial() {
    %c0_i32_ = hw.constant 0 : i32
    seq.yield %c0_i32_ : i32
  } : () -> !seq.immutable<i32>

  %init_0 = seq.initial() {
    %c0_i1_ = hw.constant 0 : i1
    seq.yield %c0_i1_ : i1
  } : () -> !seq.immutable<i1>

  %c1_i1 = hw.constant 1: i1 
  %c0_i1 = hw.constant 1: i1 
  %ce_next = comb.xor bin %ce, %c1_i1 : i1 
  %ce = seq.compreg %ce_next, %clk reset %rst, %c0_i1 initial %init_0 : i1
  
  %reg = seq.compreg.ce %next, %clk, %ce reset %rst, %c0_i32 initial %init : i32
  %next = comb.add %reg, %c1_i32 : i32

  %pa = ltl.past %reg, 5 clk %clk_0 : i32
  %m5 = comb.sub %reg, %c5_i32 : i32

  %pred = comb.icmp bin eq %m5, %pa : i32
  verif.assert %pred : i1
}

// CHECK: 1 sort bitvec 1
// CHECK-NEXT: 2 input 1 rst
// CHECK-NEXT: 3 constd 1 0
// CHECK-NEXT: 4 state 1 ce
// CHECK-NEXT: 5 init 1 4 3
// CHECK-NEXT: 6 sort bitvec 32
// CHECK-NEXT: 7 constd 6 0
// CHECK-NEXT: 8 state 6 reg
// CHECK-NEXT: 9 init 6 8 7
// CHECK-NEXT: 10 state 6 _sh1
// CHECK-NEXT: 11 state 6 _sh2
// CHECK-NEXT: 12 state 6 _sh3
// CHECK-NEXT: 13 state 6 _sh4
// CHECK-NEXT: 14 state 6 _sh5
// CHECK-NEXT: 15 constd 6 -5
// CHECK-NEXT: 16 constd 1 -1
// CHECK-NEXT: 17 constd 6 1
// CHECK-NEXT: 18 constd 6 0
// CHECK-NEXT: 19 xor 1 4 16
// CHECK-NEXT: 20 add 6 8 17
// CHECK-NEXT: 21 ite 6 4 20 8
// CHECK-NEXT: 22 constd 1 -1
// CHECK-NEXT: 23 add 6 8 15
// CHECK-NEXT: 24 eq 1 23 14
// CHECK-NEXT: 25 not 1 24
// CHECK-NEXT: 26 bad 25
// CHECK-NEXT: 27 ite 1 2 16 19
// CHECK-NEXT: 28 next 1 4 27
// CHECK-NEXT: 29 ite 6 2 18 21
// CHECK-NEXT: 30 next 6 8 29
// CHECK-NEXT: 31 next 6 10 8
// CHECK-NEXT: 32 next 6 11 10
// CHECK-NEXT: 33 next 6 12 11
// CHECK-NEXT: 34 next 6 13 12
// CHECK-NEXT: 35 next 6 14 13
