// RUN: arcilator %s | FileCheck %s

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

// CHECK-LABEL: hw.module @Top
hw.module @Top(%clock: i1, %i0: i4, %i1: i4) -> (out: i4) {
  // CHECK-NOT: hw.instance
  // CHECK-DAG: [[T0:%.+]] = arc.state @[[ADD_ARC]](%i0, %i1) lat 0
  %0 = comb.add %i0, %i1 : i4
  // CHECK-DAG: [[T3:%.+]] = arc.state @[[XOR_ARC]]([[T0]], %i0) clock %clock lat 1
  // CHECK-DAG: [[T4:%.+]] = arc.state @[[XOR_ARC]]([[T0]], %i1) clock %clock lat 1
  %1 = comb.xor %0, %i0 : i4
  %2 = comb.xor %0, %i1 : i4
  %3 = seq.compreg %1, %clock : i4
  %4 = seq.compreg %2, %clock : i4
  // CHECK-DAG: [[T5:%.+]] = arc.state @[[MUL_ARC]]([[T3]], [[T4]]) lat 0
  %5 = comb.mul %3, %4 : i4
  // CHECK-DAG: [[K:%.+]] = hw.constant 6 :
  // CHECK-DAG: [[T6:%.+]] = arc.state @[[ADD_ARC]]([[T5]], [[K]]) clock %clock lat 1
  %6 = hw.instance "child" @Child(clock: %clock: i1, a: %5: i4) -> (z: i4)
  // CHECK-DAG: hw.output [[T6]]
  hw.output %6 : i4
}

// CHECK-NOT: hw.module private @Child
hw.module private @Child(%clock: i1, %a: i4) -> (z: i4) {
  %c6_i4 = hw.constant 6 : i4
  %0 = comb.add %a, %c6_i4 : i4
  %1 = seq.compreg %0, %clock : i4
  hw.output %1 : i4
}
