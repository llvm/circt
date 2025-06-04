// RUN: circt-opt %s -canonicalize | FileCheck %s

// CHECK-LABEL: hw.module @EmptyProcess(
hw.module @EmptyProcess() {
  llhd.process {
    llhd.halt
  }
  // CHECK-NEXT: hw.output
}

// CHECK-LABEL: hw.module @DontRemoveEmptyProcessWithResults(
hw.module @DontRemoveEmptyProcessWithResults(in %a: i42, out z: i42) {
  // CHECK-NEXT: llhd.process -> i42 {
  // CHECK-NEXT:   llhd.halt
  // CHECK-NEXT: }
  // CHECK-NEXT: hw.output
  %0 = llhd.process -> i42 {
    llhd.halt %a : i42
  }
  hw.output %0 : i42
}
