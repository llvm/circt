// RUN: circt-opt %s | circt-opt | FileCheck %s

%true = hw.constant true
%s = unrealized_conversion_cast to !ltl.sequence
%p = unrealized_conversion_cast to !ltl.property

//===----------------------------------------------------------------------===//
// Assertions
//===----------------------------------------------------------------------===//

// CHECK: verif.assert {{%.+}} : i1
// CHECK: verif.assert {{%.+}} label "foo1" : i1
// CHECK: verif.assert {{%.+}} : !ltl.sequence
// CHECK: verif.assert {{%.+}} : !ltl.property
verif.assert %true : i1
verif.assert %true label "foo1" : i1
verif.assert %s : !ltl.sequence
verif.assert %p : !ltl.property

// CHECK: verif.assume {{%.+}} : i1
// CHECK: verif.assume {{%.+}} label "foo2" : i1
// CHECK: verif.assume {{%.+}} : !ltl.sequence
// CHECK: verif.assume {{%.+}} : !ltl.property
verif.assume %true : i1
verif.assume %true label "foo2" : i1
verif.assume %s : !ltl.sequence
verif.assume %p : !ltl.property

// CHECK: verif.cover {{%.+}} : i1
// CHECK: verif.cover {{%.+}} label "foo3" : i1
// CHECK: verif.cover {{%.+}} : !ltl.sequence
// CHECK: verif.cover {{%.+}} : !ltl.property
verif.cover %true : i1
verif.cover %true label "foo3" : i1
verif.cover %s : !ltl.sequence
verif.cover %p : !ltl.property


//===----------------------------------------------------------------------===//
// Print-related
// Must be inside hw.module to ensure that the dialect is loaded.
//===----------------------------------------------------------------------===//

hw.module @foo() {
// CHECK:    %false = hw.constant false
// CHECK:    %[[FSTR:.*]] = verif.format_verilog_string "Hi %x\0A"(%false) : i1
// CHECK:    verif.print %[[FSTR]]
  %false = hw.constant false
  %fstr = verif.format_verilog_string "Hi %x\0A" (%false) : i1
  verif.print %fstr
}

// CHECK-LABEL: hw.module @HasBeenReset
hw.module @HasBeenReset(in %clock: i1, in %reset: i1) {
  // CHECK-NEXT: verif.has_been_reset %clock, async %reset
  // CHECK-NEXT: verif.has_been_reset %clock, sync %reset
  %hbr0 = verif.has_been_reset %clock, async %reset
  %hbr1 = verif.has_been_reset %clock, sync %reset
}

//===----------------------------------------------------------------------===//
// Logical Equivalence Checking related operations
//===----------------------------------------------------------------------===//

// CHECK: verif.lec first {
// CHECK: } second {
// CHECK: }
verif.lec first {
} second {
}

// CHECK: verif.lec {verif.some_attr} first {
// CHECK: ^bb0(%{{.*}}: i32, %{{.*}}: i32):
// CHECK:   verif.yield %{{.*}}, %{{.*}} : i32, i32 {verif.some_attr}
// CHECK: } second {
// CHECK: ^bb0(%{{.*}}: i32, %{{.*}}: i32):
// CHECK:   verif.yield %{{.*}}, %{{.*}} : i32, i32 {verif.some_attr}
// CHECK: }
verif.lec {verif.some_attr} first {
^bb0(%arg0: i32, %arg1: i32):
  verif.yield %arg0, %arg1 : i32, i32 {verif.some_attr}
} second {
^bb0(%arg0: i32, %arg1: i32):
  verif.yield %arg0, %arg1 : i32, i32 {verif.some_attr}
}

verif.bmc bound 10 attributes {verif.some_attr}
init {
  %c0_i1 = hw.constant 0 : i1
  %clk = seq.to_clock %c0_i1
  %arg0 = smt.declare_fun : !smt.bv<32>
  %state0 = smt.declare_fun : !smt.bv<32>
  verif.yield %clk : !seq.clock
}
loop {
  ^bb0(%clk: !seq.clock, %arg0: i32, %state0: i32):
  %from_clock = seq.from_clock %clk
  %c-1_i1 = hw.constant -1 : i1
  %neg_clock = comb.xor %from_clock, %c-1_i1 : i1
  %newclk = seq.to_clock %neg_clock
  verif.yield %newclk : !seq.clock
}
circuit {
^bb0(%clk: !seq.clock, %arg0: i32, %state0: i32):
  %c-1_i32 = hw.constant -1 : i32
  %0 = comb.add %arg0, %state0 : i32
  // %state0 is the result of a seq.compreg taking %0 as input
  %2 = comb.xor %state0, %c-1_i32 : i32
  verif.yield %2, %0 : i32, i32
}
