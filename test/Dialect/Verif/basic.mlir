// RUN: circt-opt --verify-roundtrip %s | FileCheck %s

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
// Formal Test
//===----------------------------------------------------------------------===//

// CHECK-LABEL: verif.formal @EmptyFormalTest
verif.formal @EmptyFormalTest {} {
}

// CHECK-LABEL: verif.formal @FormalTestWithAttrs
verif.formal @FormalTestWithAttrs {
  // CHECK-SAME: a,
  a,
  // CHECK-SAME: b = "hello"
  b = "hello",
  // CHECK-SAME: c = 42 : i64
  c = 42 : i64,
  // CHECK-SAME: d = ["x", "y"]
  d = ["x", "y"],
  // CHECK-SAME: e = {u, v}
  e = {u, v}
} {
}

// CHECK-LABEL: verif.formal @FormalTestBody
verif.formal @FormalTestBody {} {
  // CHECK: {{%.+}} = verif.symbolic_value : i42
  %0 = verif.symbolic_value : i42
}

//===----------------------------------------------------------------------===//
// Simulation Test
//===----------------------------------------------------------------------===//

verif.simulation @EmptySimulationTest {} {
^bb0(%clock: !seq.clock, %init: i1):
  %0 = hw.constant true
  verif.yield %0, %0 : i1, i1
}

//===----------------------------------------------------------------------===//
// Contracts
//===----------------------------------------------------------------------===//

// CHECK: [[A:%.+]] = unrealized_conversion_cast to i42
%a = unrealized_conversion_cast to i42
// CHECK: [[B:%.+]] = unrealized_conversion_cast to i1337
%b = unrealized_conversion_cast to i1337
// CHECK: [[C:%.+]] = unrealized_conversion_cast to i9001
%c = unrealized_conversion_cast to i9001
// CHECK: [[CLOCK:%.+]] = unrealized_conversion_cast to !seq.clock
%d = unrealized_conversion_cast to !seq.clock

// CHECK: verif.contract {
verif.contract {}
// CHECK: {{%.+}} = verif.contract [[A]] : i42 {
%q0 = verif.contract %a : i42 {}
// CHECK: {{%.+}}:3 = verif.contract [[A]], [[B]], [[C]] : i42, i1337, i9001 {
%q1:3 = verif.contract %a, %b, %c : i42, i1337, i9001 {}

verif.contract {
  // CHECK: verif.require {{%.+}} : i1
  // CHECK: verif.require {{%.+}} if {{%.+}} : i1
  // CHECK: verif.require {{%.+}} label "foo" : i1
  // CHECK: verif.require {{%.+}} : !ltl.sequence
  // CHECK: verif.require {{%.+}} : !ltl.property
  verif.require %true : i1
  verif.require %true if %true : i1
  verif.require %true label "foo" : i1
  verif.require %s : !ltl.sequence
  verif.require %p : !ltl.property

  // CHECK: verif.ensure {{%.+}} : i1
  // CHECK: verif.ensure {{%.+}} if {{%.+}} : i1
  // CHECK: verif.ensure {{%.+}} label "foo" : i1
  // CHECK: verif.ensure {{%.+}} : !ltl.sequence
  // CHECK: verif.ensure {{%.+}} : !ltl.property
  verif.ensure %true : i1
  verif.ensure %true if %true : i1
  verif.ensure %true label "foo" : i1
  verif.ensure %s : !ltl.sequence
  verif.ensure %p : !ltl.property
}

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

//===----------------------------------------------------------------------===//
// Refinement Checking related operations
//===----------------------------------------------------------------------===//

// CHECK: verif.refines first {
// CHECK: } second {
// CHECK: }
verif.refines first {
} second {
}

// CHECK: verif.refines {verif.some_attr} first {
// CHECK: ^bb0(%{{.*}}: i32, %{{.*}}: i32):
// CHECK:   verif.yield %{{.*}}, %{{.*}} : i32, i32 {verif.some_attr}
// CHECK: } second {
// CHECK: ^bb0(%{{.*}}: i32, %{{.*}}: i32):
// CHECK:   verif.yield %{{.*}}, %{{.*}} : i32, i32 {verif.some_attr}
// CHECK: }
verif.refines {verif.some_attr} first {
^bb0(%arg0: i32, %arg1: i32):
  verif.yield %arg0, %arg1 : i32, i32 {verif.some_attr}
} second {
^bb0(%arg0: i32, %arg1: i32):
  verif.yield %arg0, %arg1 : i32, i32 {verif.some_attr}
}

//===----------------------------------------------------------------------===//
// Bounded Model Checking related operations
//===----------------------------------------------------------------------===//

// CHECK: verif.bmc bound 10 num_regs 0 initial_values [] attributes {verif.some_attr} init {
// CHECK: } loop {
// CHECK: } circuit {
// CHECK: ^bb0(%{{.*}}):
// CHECK: verif.yield %{{.*}} : i32
// CHECK: }
verif.bmc bound 10 num_regs 0 initial_values [] attributes {verif.some_attr} init {
} loop {
} circuit {
^bb0(%arg0: i32):
  %false = hw.constant false
  // Arbitrary assertion so op verifies
  verif.assert %false : i1
  verif.yield %arg0 : i32
}

//CHECK: verif.bmc bound 10 num_regs 1 initial_values [unit] attributes {verif.some_attr}
//CHECK: init {
//CHECK:   %{{.*}} = hw.constant false
//CHECK:   %{{.*}} = seq.to_clock %{{.*}}
//CHECK:   verif.yield %{{.*}}, %{{.*}} : !seq.clock, i1
//CHECK: }
//CHECK: loop {
//CHECK:   ^bb0(%{{.*}}: !seq.clock, %{{.*}}: i1):
//CHECK:   %{{.*}} = seq.from_clock %{{.*}}
//CHECK:   %{{.*}} = hw.constant true
//CHECK:   %{{.*}} = comb.xor %{{.*}}, %{{.*}} : i1
//CHECK:   %{{.*}} = comb.xor %{{.*}}, %{{.*}} : i1
//CHECK:   %{{.*}} = seq.to_clock %{{.*}}
//CHECK:   verif.yield %{{.*}}, %{{.*}} : !seq.clock, i1
//CHECK: }
//CHECK: circuit {
//CHECK: ^bb0(%{{.*}}: !seq.clock, %{{.*}}: i32, %{{.*}}: i32):
//CHECK:   %{{.*}} = hw.constant -1 : i32
//CHECK:   %{{.*}} = comb.add %{{.*}}, %{{.*}} : i32
//CHECK:   %{{.*}} = comb.xor %{{.*}}, %{{.*}} : i32
//CHECK:   verif.yield %{{.*}}, %{{.*}} : i32, i32
//CHECK: }
verif.bmc bound 10 num_regs 1 initial_values [unit] attributes {verif.some_attr}
init {
  %c0_i1 = hw.constant 0 : i1
  %clk = seq.to_clock %c0_i1
  verif.yield %clk, %c0_i1 : !seq.clock, i1
}
loop {
  ^bb0(%clk: !seq.clock, %stateArg: i1):
  %from_clock = seq.from_clock %clk
  %c-1_i1 = hw.constant -1 : i1
  %neg_clock = comb.xor %from_clock, %c-1_i1 : i1
  %newStateArg = comb.xor %stateArg, %c-1_i1 : i1
  %newclk = seq.to_clock %neg_clock
  verif.yield %newclk, %newStateArg : !seq.clock, i1
}
circuit {
^bb0(%clk: !seq.clock, %arg0: i32, %state0: i32):
  %c-1_i32 = hw.constant -1 : i32
  %0 = comb.add %arg0, %state0 : i32
  // %state0 is the result of a seq.compreg taking %0 as input
  %2 = comb.xor %state0, %c-1_i32 : i32
  %false = hw.constant false
  // Arbitrary assertion so op verifies
  verif.assert %false : i1
  verif.yield %2, %0 : i32, i32
}
