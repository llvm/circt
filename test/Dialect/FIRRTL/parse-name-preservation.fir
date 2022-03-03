; RUN: firtool --parse-only %s | FileCheck %s
circuit Bar :
  ; NamePreservation should create attaches for analog typed elements.
  ; https://github.com/llvm/circt/issues/2718
  module Bar :
    input a: {flip a: UInt<1>, b: Analog<1>}[1]
    output b: {flip a: UInt<1>, b: Analog<1>}[1]

    wire w: {flip a: UInt<1>, b: Analog<1>}[1]
    ; CHECK: %_w = firrtl.wire
    ; CHECK: [[_W_0:%.+]] = firrtl.subindex %_w[0]
    ; CHECK: [[_W_0_1:%.+]] = firrtl.subfield [[_W_0]](1)
    ; CHECK: [[_W_0_0:%.+]] = firrtl.subfield [[_W_0]](0)
    ; CHECK: %w = firrtl.wire sym @w
    ; CHECK: [[W_0:%.+]] = firrtl.subindex %w[0]
    ; CHECK: [[W_0_1:%.+]] = firrtl.subfield [[W_0]](1)
    ; CHECK: [[W_0_0:%.+]] = firrtl.subfield [[W_0]](0)
    ; CHECK: firrtl.connect [[W_0_0]], [[_W_0_0]]
    ; CHECK: firrtl.attach [[W_0_1]], [[_W_0_1]]
    a[0].a <= w[0].a
    w[0].a <= b[0].a
    attach(a[0].b, w[0].b, b[0].b)