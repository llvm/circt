// RUN: circt-opt %s -canonicalize | FileCheck %s

// CHECK-LABEL: hw.module @EmptyProcess(
hw.module @EmptyProcess() {
  llhd.process {
    llhd.halt
  }
// CHECK-NEXT: hw.output
 hw.output
}
