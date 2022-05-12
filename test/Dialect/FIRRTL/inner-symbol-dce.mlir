// RUN: circt-opt -pass-pipeline='firrtl.circuit(inner-symbol-dce)' %s | FileCheck %s

// CHECK-LABEL: firrtl.circuit "Simple"
firrtl.circuit "Simple" attributes {
  annotations = [
    {
      class = "circuit",
      key = #hw.innerNameRef<@Simple::@w0>,
      dict = {key = #hw.innerNameRef<@Simple::@w1>},
      array = [#hw.innerNameRef<@Simple::@w2>],
      payload = "hello"
    }
  ]} {
  // CHECK: firrtl.module @Simple
  firrtl.module @Simple() {
    // CHECK-NEXT: @w0
    %w0 = firrtl.wire sym @w0 {inner_sym_visibility = "private"} : !firrtl.uint<1>
    // CHECK-NEXT: @w1
    %w1 = firrtl.wire sym @w1 {inner_sym_visibility = "private"} : !firrtl.uint<1>
    // CHECK-NEXT: @w2
    %w2 = firrtl.wire sym @w2 {inner_sym_visibility = "private"} : !firrtl.uint<1>
    // CHECK-NEXT: %w3
    // CHECK-NOT: @w3
    %w3 = firrtl.wire sym @w3 {inner_sym_visibility = "private"} : !firrtl.uint<1>
    // CHECK-NEXT: @w4
    %w4 = firrtl.wire sym @w4 : !firrtl.uint<1>
  }
}
