// RUN: circt-opt -pass-pipeline='builtin.module(firrtl-inner-symbol-dce)' %s | FileCheck %s

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

  // CHECK-LABEL: firrtl.module @Simple
  firrtl.module @Simple() {
    // CHECK-NEXT: @w0
    %w0 = firrtl.wire sym @w0 : !firrtl.uint<1>
    // CHECK-NEXT: @w1
    %w1 = firrtl.wire sym @w1 : !firrtl.uint<1>
    // CHECK-NEXT: @w2
    %w2 = firrtl.wire sym @w2 : !firrtl.uint<1>
    // CHECK-NEXT: @w3
    %w3 = firrtl.wire sym @w3 : !firrtl.uint<1>
    // CHECK-NEXT: %w4
    // CHECK-NOT:  @w4
    %w4 = firrtl.wire sym @w4 : !firrtl.uint<1>

    firrtl.instance child sym @child @Child()
  }

  // CHECK-LABEL: firrtl.module @Child
  firrtl.module @Child() {
    // CHECK-NEXT: @w5
    %w5 = firrtl.wire sym @w5 : !firrtl.uint<1>
  }

  firrtl.hierpath private @nla [@Simple::@child, @Child::@w5]
}

sv.verbatim "{{0}}" {symbols = [#hw.innerNameRef<@Simple::@w3>]}
