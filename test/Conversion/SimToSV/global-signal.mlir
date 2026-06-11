// RUN: circt-opt --lower-sim-to-sv %s | FileCheck %s

sim.global_signal @STOP_COND_ : i1 {
  %true = hw.constant true
  sim.yield %true : i1
}

sim.global_signal @PRINTF_COND_ : i1 {
  %stop = sim.global_signal.read @STOP_COND_ : i1
  %true = hw.constant true
  %not_stop = comb.xor %stop, %true : i1
  sim.yield %not_stop : i1
}

sim.global_signal @COMPLEX_COND_ : i4 {
  %a = hw.constant 10 : i4
  %b = hw.constant 12 : i4
  %cmp = comb.icmp ult %a, %b : i4
  %mux = comb.mux %cmp, %a, %b : i4
  %slice = comb.extract %mux from 1 : (i4) -> i2
  %rep = comb.replicate %slice : (i2) -> i4
  sim.yield %rep : i4
}

// CHECK:      sv.macro.decl @STOP_COND_
// CHECK-NEXT: emit.fragment @STOP_COND__FRAGMENT {
// CHECK-NEXT:   sv.ifdef @STOP_COND_ {
// CHECK-NEXT:   } else {
// CHECK-NEXT:     sv.macro.def @STOP_COND_ "1'h1"
// CHECK-NEXT:   }
// CHECK-NEXT: }
// CHECK-NEXT: sv.macro.decl @PRINTF_COND_
// CHECK-NEXT: emit.fragment @PRINTF_COND__FRAGMENT {
// CHECK-NEXT:   sv.ifdef @PRINTF_COND_ {
// CHECK-NEXT:   } else {
// CHECK-NEXT:     sv.macro.def @PRINTF_COND_ "~(`STOP_COND_)"
// CHECK-NEXT:   }
// CHECK-NEXT: }
// CHECK-NEXT: sv.macro.decl @COMPLEX_COND_
// CHECK-NEXT: emit.fragment @COMPLEX_COND__FRAGMENT {
// CHECK-NEXT:   sv.ifdef @COMPLEX_COND_ {
// CHECK-NEXT:   } else {
// CHECK-NEXT:     sv.macro.def @COMPLEX_COND_ "{2{(((4'hA) < (4'hC)) ? (4'hA) : (4'hC))[2:1]}}"
// CHECK-NEXT:   }
// CHECK-NEXT: }

// CHECK-LABEL: hw.module @read_stop
// CHECK-SAME: attributes {emit.fragments = [@STOP_COND__FRAGMENT]}
hw.module @read_stop(out stop: i1) {
  // CHECK: %[[STOP:.+]] = sv.macro.ref.expr.se @STOP_COND_() : () -> i1
  %stop = sim.global_signal.read @STOP_COND_ : i1
  hw.output %stop : i1
}

// CHECK-LABEL: hw.module @read_printf
// CHECK-SAME: attributes {emit.fragments = [@PRINTF_COND__FRAGMENT, @STOP_COND__FRAGMENT]}
hw.module @read_printf(out printf: i1) {
  // CHECK: %[[PRINTF:.+]] = sv.macro.ref.expr.se @PRINTF_COND_() : () -> i1
  %printf = sim.global_signal.read @PRINTF_COND_ : i1
  hw.output %printf : i1
}

// CHECK-LABEL: hw.module @read_complex
// CHECK-SAME: attributes {emit.fragments = [@COMPLEX_COND__FRAGMENT]}
hw.module @read_complex(out complex: i4) {
  // CHECK: %[[COMPLEX:.+]] = sv.macro.ref.expr.se @COMPLEX_COND_() : () -> i4
  %complex = sim.global_signal.read @COMPLEX_COND_ : i4
  hw.output %complex : i4
}
