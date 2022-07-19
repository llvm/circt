// RUN: circt-opt %s | FileCheck %s

// CHECK-LABEL: @ctor
hw.module @ctor () {
  //CHECK-NEXT: systemc.ctor {
  systemc.ctor {
  //CHECK-NEXT: }
  }
}
