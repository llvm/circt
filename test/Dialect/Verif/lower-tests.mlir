// RUN: circt-opt --verif-lower-tests %s | FileCheck %s

// CHECK-LABEL: hw.module @SomeFormalTest(
// CHECK-SAME: in [[SYM0:%.+]] : i42
// CHECK-SAME: in [[SYM1:%.+]] : i9001
// CHECK-SAME: comment = "FORMAL TEST: SomeFormalTest {foo = \22bar\22}"
// CHECK-NOT: verif.formal
verif.formal @SomeFormalTest {foo = "bar"} {
  // CHECK-NOT: verif.symbolic_value
  %0 = verif.symbolic_value : i42
  %1 = verif.symbolic_value : i9001
  // CHECK-NEXT: dbg.variable "a", [[SYM0]]
  dbg.variable "a", %0 : i42
  // CHECK-NEXT: dbg.variable "b", [[SYM1]]
  dbg.variable "b", %1 : i9001
}

// CHECK-LABEL: hw.module @SomeSimulationTest(
// CHECK-SAME: in %clock : !seq.clock
// CHECK-SAME: in %init : i1
// CHECK-SAME: out done : i1
// CHECK-SAME: out success : i1
// CHECK-SAME: comment = "SIMULATION TEST: SomeSimulationTest {foo = \22bar\22}"
// CHECK-NOT: verif.simulation
verif.simulation @SomeSimulationTest {foo = "bar"} {
^bb0(%clock: !seq.clock, %init: i1):
  // CHECK-NEXT: dbg.variable "a", %clock
  dbg.variable "a", %clock : !seq.clock
  // CHECK-NEXT: dbg.variable "b", %init
  dbg.variable "b", %init : i1
  // CHECK-NEXT: [[C:%.+]] = builtin.unrealized_conversion_cast to i1 {c}
  %0 = builtin.unrealized_conversion_cast to i1 {c}
  // CHECK-NEXT: [[D:%.+]] = builtin.unrealized_conversion_cast to i1 {d}
  %1 = builtin.unrealized_conversion_cast to i1 {d}
  // CHECK-NEXT: hw.output [[C]], [[D]] : i1, i1
  // CHECK-NOT: verif.yield
  verif.yield %0, %1 : i1, i1
}
