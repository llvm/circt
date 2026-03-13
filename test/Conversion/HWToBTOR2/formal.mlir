// RUN: circt-opt %s --convert-hw-to-btor2 -o %t | FileCheck %s  

module {
  // CHECK: [[I32:[0-9]+]] sort bitvec 32
  // CHECK: [[A:[0-9]+]] input [[I32]] a
  // CHECK: [[B:[0-9]+]] input [[I32]] b
  // CHECK: [[C:[0-9]+]] input [[I32]] c
  // CHECK: [[ADD:[0-9]+]] add [[I32]] [[A]] [[B]]
  // CHECK: [[I1:[0-9]+]] sort bitvec 1
  // CHECK: [[RES:[0-9]+]] eq [[I1]] [[ADD]] [[C]]
  // CHECK: [[NOT:[0-9]+]] not [[I1]] [[RES]]
  // CHECK: [[BAD:[0-9]+]] bad [[NOT]]
  verif.formal @foo {} {
    %a = verif.symbolic_value : i32
    %b = verif.symbolic_value : i32
    %c = verif.symbolic_value : i32
    %res = comb.add %a, %b : i32 
    %cond = comb.icmp eq %res, %c : i32
    verif.assert %cond : i1
  }

  /// should work alongside hw.module ops

  // CHECK: [[I32_:[0-9]+]] sort bitvec 32
  // CHECK: [[A_:[0-9]+]] input [[I32_]] a
  // CHECK: [[B_:[0-9]+]] input [[I32_]] b
  // CHECK: [[C_:[0-9]+]] input [[I32_]] c
  // CHECK: [[ADD_:[0-9]+]] add [[I32_]] [[A_]] [[B_]]
  // CHECK: [[I1_:[0-9]+]] sort bitvec 1
  // CHECK: [[RES_:[0-9]+]] eq [[I1_]] [[ADD_]] [[C_]]
  // CHECK: [[NOT_:[0-9]+]] not [[I1_]] [[RES_]]
  // CHECK: [[BAD_:[0-9]+]] bad [[NOT_]]
  hw.module @bar(in %a : i32) {
    %b = verif.symbolic_value : i32
    %c = verif.symbolic_value : i32
    %res = comb.add %a, %b : i32 
    %cond = comb.icmp eq %res, %c : i32
    verif.assert %cond : i1
  }
}