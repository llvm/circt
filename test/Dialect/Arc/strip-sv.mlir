// RUN: circt-opt %s --arc-strip-sv | FileCheck %s

// CHECK-NOT: sv.verbatim
// CHECK-NOT: sv.ifdef
sv.verbatim "// Standard header to adapt well known macros to our needs." {symbols = []}
sv.ifdef  "RANDOMIZE_REG_INIT" {
  sv.verbatim "`define RANDOMIZE" {symbols = []}
}

// CHECK-LABEL: hw.module @Foo(
hw.module @Foo(%clock: i1, %a: i4) -> (z: i4) {
  // CHECK-NEXT: [[REG:%.+]] = seq.compreg %a, %clock
  %0 = seq.firreg %a clock %clock : i4
  %1 = sv.wire : !hw.inout<i4>
  sv.assign %1, %0 : i4
  %2 = sv.read_inout %1 : !hw.inout<i4>
  // CHECK-NEXT: hw.output [[REG]]
  hw.output %2 : i4
}
// CHECK-NEXT: }
