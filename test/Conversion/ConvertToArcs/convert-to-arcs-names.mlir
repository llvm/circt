// RUN: circt-opt %s --convert-to-arcs="tap-registers=0" | FileCheck %s --check-prefixes=CHECK,CHECK-TAP-OFF
// RUN: circt-opt %s --convert-to-arcs="tap-registers=1" | FileCheck %s --check-prefixes=CHECK,CHECK-TAP-ON

// CHECK-LABEL: hw.module @Trivial(
hw.module @Trivial(in %clock: !seq.clock, in %i0: i4, in %reset: i1, out o0: i4) {
  // CHECK: arc.state {{@.+}}(%i0) clock %clock lat 1
  // CHECK-TAP-OFF-NOT: names = ["foo"]
  // CHECK-TAP-ON: names = ["foo"]
  %foo = seq.compreg %i0, %clock : i4
  hw.output %foo : i4
}
