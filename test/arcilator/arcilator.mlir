// RUN: arcilator %s --inline=0 --until-before=state-alloc | FileCheck %s
// RUN: arcilator %s | FileCheck %s --check-prefix=LLVM
// RUN: arcilator --print-debug-info %s | FileCheck %s --check-prefix=LLVM-DEBUG

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
hw.module @Top(in %clock : !seq.clock, in %i0 : i4, in %i1 : i4, out out : i4) {
  // CHECK-DAG: [[CLOCK:%.+]] = arc.root_input "clock", %arg0 {{.+}} !arc.state<i1>
  // CHECK-DAG: [[I0:%.+]] = arc.root_input "i0", %arg0 {{.+}} !arc.state<i4>
  // CHECK-DAG: [[I1:%.+]] = arc.root_input "i1", %arg0 {{.+}} !arc.state<i4>
  // CHECK-DAG: [[PTR_OUT:%.+]] = arc.root_output "out", %arg0 {{.+}} !arc.state<i4>
  // CHECK-DAG: [[CLOCK_OLD:%.+]] = arc.alloc_state %arg0 {{.+}} !arc.state<i1>
  // CHECK-DAG: [[FOO:%.+]] = arc.alloc_state %arg0 {name = "foo"} {{.+}} !arc.state<i4>
  // CHECK-DAG: [[BAR:%.+]] = arc.alloc_state %arg0 {name = "bar"} {{.+}} !arc.state<i4>

  // CHECK-DAG: [[READ_CLOCK:%.+]] = arc.state_read [[CLOCK]]
  // CHECK-DAG: arc.state_write [[CLOCK_OLD]] = [[READ_CLOCK]]
  // CHECK-DAG: [[READ_CLOCK_OLD:%.+]] = arc.state_read [[CLOCK_OLD]]
  // CHECK-DAG: [[CLOCK_CHANGED:%.+]] = comb.icmp ne [[READ_CLOCK_OLD]], [[READ_CLOCK]] : i1
  // CHECK-DAG: [[CLOCK_ROSE:%.+]] = comb.and [[CLOCK_CHANGED]], [[READ_CLOCK]] : i1

  // CHECK-DAG: arc.clock_tree [[CLOCK_ROSE]] {
  // CHECK-DAG:   [[READ_I0:%.+]] = arc.state_read [[I0]]
  // CHECK-DAG:   [[READ_I1:%.+]] = arc.state_read [[I1]]
  // CHECK-DAG:   [[ADD:%.+]] = arc.state @[[ADD_ARC]]([[READ_I0]], [[READ_I1]]) lat 0
  // CHECK-DAG:   [[XOR1:%.+]] = arc.state @[[XOR_ARC]]([[ADD]], [[READ_I0]]) lat 0
  // CHECK-DAG:   [[XOR2:%.+]] = arc.state @[[XOR_ARC]]([[ADD]], [[READ_I1]]) lat 0
  // CHECK-DAG:   arc.state_write [[FOO]] = [[XOR1]]
  // CHECK-DAG:   arc.state_write [[BAR]] = [[XOR2]]
  // CHECK-DAG: }

  // CHECK-DAG: arc.passthrough {
  // CHECK-DAG:   [[READ_FOO:%.+]] = arc.state_read [[FOO]]
  // CHECK-DAG:   [[READ_BAR:%.+]] = arc.state_read [[BAR]]
  // CHECK-DAG:   [[MUL:%.+]] = arc.state @[[MUL_ARC]]([[READ_FOO]], [[READ_BAR]]) lat 0
  // CHECK-DAG:   arc.state_write [[PTR_OUT]] = [[MUL]]
  // CHECK-DAG: }

  %0 = comb.add %i0, %i1 : i4
  %1 = comb.xor %0, %i0 : i4
  %2 = comb.xor %0, %i1 : i4
  %foo = seq.compreg %1, %clock : i4
  %bar = seq.compreg %2, %clock : i4
  %3 = comb.mul %foo, %bar : i4
  hw.output %3 : i4
}

// LLVM: define void @Top_passthrough(ptr %0)
// LLVM:   mul i4
// LLVM: define void @Top_clock(ptr %0)
// LLVM:   add i4
// LLVM:   xor i4
// LLVM:   xor i4

// LLVM-DEBUG: define void @Top_passthrough(ptr %0){{.*}}!dbg
// LLVM-DEBUG:   mul i4{{.*}}!dbg
// LLVM-DEBUG: define void @Top_clock(ptr %0){{.*}}!dbg
// LLVM-DEBUG:   add i4{{.*}}!dbg
// LLVM-DEBUG:   xor i4{{.*}}!dbg
// LLVM-DEBUG:   xor i4{{.*}}!dbg
