// RUN: arcilator %s --inline=0 --until-before=state-lowering | FileCheck %s

// CHECK-LABEL: module {
// CHECK-DAG:   arc.define @[[ARC:.+]]([[ARG0:%.+]]: i4)
// CHECK-DAG:     arc.output [[ARG0]]
// CHECK-NEXT:  }

// CHECK-DAG: hw.module @Trivial([[CLOCK:%.+]]: i1, [[I0:%.+]]: i4, [[RESET:%.+]]: i1)
hw.module @Trivial(%clock: i1, %i0: i4, %reset: i1) -> (out: i4) {
  // CHECK-DAG: [[RES0:%.+]] = arc.state @[[ARC]]([[I0]]) clock [[CLOCK]] reset [[RESET]] lat 1 {names = ["foo"]
  // CHECK-DAG: hw.output [[RES0:%.+]]
  %0 = hw.constant 0 : i4
  %foo = seq.compreg %i0, %clock, %reset, %0 : i4
  hw.output %foo : i4
}
// CHECK-DAG: }

// CHECK-DAG: hw.module @NonTrivial([[CLOCK]]: i1, [[I0_2:%.+]]: i4, [[RESET1:%.+]]: i1, [[RESET2:%.+]]: i1)
hw.module @NonTrivial(%clock: i1, %i0: i4, %reset1: i1, %reset2: i1) -> (out1: i4, out2: i4) {
  // CHECK-DAG: [[RES2:%.+]] = arc.state @[[ARC]]([[I0_2]]) clock [[CLOCK]] reset [[RESET1]] lat 1 {names = ["foo"]
  // CHECK-DAG: [[RES3:%.+]] = arc.state @[[ARC]]([[I0_2]]) clock [[CLOCK]] reset [[RESET2]] lat 1 {names = ["bar"]
  // CHECK-DAG: hw.output [[RES2]], [[RES3]]
  %0 = hw.constant 0 : i4
  %foo = seq.compreg %i0, %clock, %reset1, %0 : i4
  %bar = seq.compreg %i0, %clock, %reset2, %0 : i4
  hw.output %foo, %bar : i4, i4
}

// CHECK-DAG: }
// CHECK-DAG: }
