// RUN: circt-opt %s | circt-opt | FileCheck %s

// CHECK-LABEL: func @bitvectors
func.func @bitvectors() {
  // CHECK: %c5_bv32 = smt.bv.constant #smt.bv<5> : !smt.bv<32> {smt.some_attr}
  %c5_bv32 = smt.bv.constant #smt.bv<5> : !smt.bv<32> {smt.some_attr}
  // CHECK: %c92_bv8 = smt.bv.constant #smt.bv<92> : !smt.bv<8> {smt.some_attr}
  %c92_bv8 = smt.bv.constant #smt.bv<0x5c> : !smt.bv<8> {smt.some_attr}
  // CHECK: %c-1_bv8 = smt.bv.constant #smt.bv<-1> : !smt.bv<8>
  %c-1_bv8 = smt.bv.constant #smt.bv<-1> : !smt.bv<8>

  return
}
