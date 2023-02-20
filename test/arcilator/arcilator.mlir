// RUN: arcilator --inline=0 %s | FileCheck %s

// CHECK:      arc.define @[[XOR_ARC:.+]](
// CHECK-NEXT:   comb.xor
// CHECK-NEXT:   arc.output
// CHECK-NEXT: }

// CHECK:      arc.define @[[ADD_ARC:.+]](
// CHECK-NEXT:   comb.add
// CHECK-NEXT:   arc.output
// CHECK-NEXT: }

// CHECK:      arc.define @[[MUL_ARC:.+]](
// CHECK-NEXT:   comb.mul
// CHECK-NEXT:   arc.output
// CHECK-NEXT: }

// CHECK-NOT: hw.module @Top
// CHECK-LABEL: arc.model "Top" {
// CHECK-NEXT: ^bb0(%arg0: !arc.storage):
hw.module @Top(%clock: i1, %i0: i4, %i1: i4) -> (out: i4) {
  // CHECK-DAG: [[CLOCK:%.+]] = arc.root_input "clock"
  // CHECK-DAG: [[I0:%.+]] = arc.root_input "i0"
  // CHECK-DAG: [[I1:%.+]] = arc.root_input "i1"
  // CHECK-DAG: [[OUT:%.+]] = arc.root_output "out"

  // CHECK-DAG: [[FOO:%.+]] = arc.alloc_state %arg0 {name = "foo"}
  // CHECK-DAG: [[BAR:%.+]] = arc.alloc_state %arg0 {name = "bar"}

  // CHECK-DAG: arc.passthrough {
  // CHECK-DAG:   [[READ_FOO:%.+]] = arc.state_read [[FOO]]
  // CHECK-DAG:   [[READ_BAR:%.+]] = arc.state_read [[BAR]]
  // CHECK-DAG:   [[MUL:%.+]] = arc.state @[[MUL_ARC]]([[READ_FOO]], [[READ_BAR]]) lat 0
  // CHECK-DAG:   arc.state_write [[OUT]] = [[MUL]]
  // CHECK-DAG: }

  // CHECK-DAG: [[READ_CLOCK:%.+]] = arc.state_read [[CLOCK]]
  // CHECK-DAG:  arc.clock_tree [[READ_CLOCK]] {
  // CHECK-DAG:   [[READ_I0:%.+]] = arc.state_read [[I0]]
  // CHECK-DAG:   [[READ_I1:%.+]] = arc.state_read [[I1]]
  // CHECK-DAG:   [[ADD:%.+]] = arc.state @[[ADD_ARC]]([[READ_I0]], [[READ_I1]]) lat 0
  // CHECK-DAG:   [[XOR1:%.+]] = arc.state @[[XOR_ARC]]([[ADD]], [[READ_I0]]) lat 0
  // CHECK-DAG:   [[XOR2:%.+]] = arc.state @[[XOR_ARC]]([[ADD]], [[READ_I1]]) lat 0
  // CHECK-DAG:   arc.state_write [[FOO]] = [[XOR1]]
  // CHECK-DAG:   arc.state_write [[BAR]] = [[XOR2]]
  // CHECK-DAG:  }

  %0 = comb.add %i0, %i1 : i4
  %1 = comb.xor %0, %i0 : i4
  %2 = comb.xor %0, %i1 : i4
  %foo = seq.compreg %1, %clock : i4
  %bar = seq.compreg %2, %clock : i4
  %3 = comb.mul %foo, %bar : i4
  hw.output %3 : i4
}
