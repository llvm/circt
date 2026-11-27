// RUN: circt-opt %s --convert-hw-to-btor2 -o %t | FileCheck %s

module {
  // CHECK: [[I32:[0-9]+]] sort bitvec 32
  // CHECK: [[A:[0-9]+]] input [[I32]] a
  // CHECK: [[B:[0-9]+]] input [[I32]] b
  // CHECK: [[I1:[0-9]+]] sort bitvec 1
  // CHECK: [[EN:[0-9]+]] input [[I1]] en
  hw.module @foo(in %clk : !seq.clock, in %a : i32, in %b : i32, in %en : i1) {
    %0 = seq.from_clock %clk

    // CHECK: [[EQ:[0-9]+]] eq [[I1]] [[A]] [[B]]
    %eq = comb.icmp eq %a, %b : i32
    // CHECK: [[ULT:[0-9]+]] ult [[I1]] [[A]] [[B]]
    %ult = comb.icmp ult %a, %b : i32

    // CHECK: {{[0-9]+}} constraint [[EQ]]{{$}}
    verif.clocked_assume %eq, posedge %0 : i1

    // The enable is folded into the property as an implication.
    // CHECK: [[IMPL0:[0-9]+]] implies [[I1]] [[EN]] [[EQ]]
    // CHECK-NEXT: {{[0-9]+}} constraint [[IMPL0]]{{$}}
    verif.clocked_assume %eq if %en, posedge %0 : i1

    // CHECK: [[NOT0:[0-9]+]] not [[I1]] [[ULT]]
    // CHECK-NEXT: {{[0-9]+}} bad [[NOT0]]{{$}}
    verif.clocked_assert %ult, posedge %0 : i1

    // CHECK: [[IMPL1:[0-9]+]] implies [[I1]] [[EN]] [[ULT]]
    // CHECK-NEXT: [[NOT1:[0-9]+]] not [[I1]] [[IMPL1]]
    // CHECK-NEXT: {{[0-9]+}} bad [[NOT1]]{{$}}
    verif.clocked_assert %ult if %en, posedge %0 : i1

    hw.output
  }
}
