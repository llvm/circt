; RUN: firtool --parse-only %s | FileCheck %s
circuit Bar :
  module Bar :
    input in: UInt<1>

    ; Should create a "tap" node with the same type and a symbol when the type
    ; is passive.
    wire a: UInt<1>
    ; CHECK: %_a = firrtl.wire  : !firrtl.uint<1>
    ; CHECK: %a = firrtl.node sym @a %_a  : !firrtl.uint<1>

    ; Should use the non-tap wire in expressions.
    a <= in
    ; CHECK: firrtl.connect %_a, %in

    ; When the type is not passive, the tap should be a wire with the passive
    ; type of the original wire.
    wire flip: {flip a: UInt<1>}
    ; CHECK: %_flip = firrtl.wire  : !firrtl.bundle<a flip: uint<1>>
    ; CHECK-NEXT: [[_flip_0:%.+]] = firrtl.subfield %_flip(0)
    ; CHECK: %flip = firrtl.wire sym @flip  : !firrtl.bundle<a: uint<1>>
    ; CHECK-NEXT: [[flip_0:%.+]] = firrtl.subfield %flip(0)
    ; CHECK-NEXT: firrtl.connect [[flip_0]], [[_flip_0]]

    ; Analog values should be tapped with a node.
    wire analog: Analog<1>
    ; CHECK: %_analog = firrtl.wire  : !firrtl.analog<1>
    ; CHECK: %analog = firrtl.node sym @analog %_analog  : !firrtl.analog<1>

    ; Should create attaches for analog typed elements between the tap wire and
    ; the original wire.
    ; https://github.com/llvm/circt/issues/2718
    wire w: {flip a: UInt<1>, b: Analog<1>}[1]
    ; CHECK: %_w = firrtl.wire  : !firrtl.vector<bundle<a flip: uint<1>, b: analog<1>>, 1>
    ; CHECK: [[_W_0:%.+]] = firrtl.subindex %_w[0]
    ; CHECK: [[_W_0_1:%.+]] = firrtl.subfield [[_W_0]](1)
    ; CHECK: [[_W_0_0:%.+]] = firrtl.subfield [[_W_0]](0)
    ; CHECK: %w = firrtl.wire sym @w  : !firrtl.vector<bundle<a: uint<1>, b: analog<1>>, 1>
    ; CHECK: [[W_0:%.+]] = firrtl.subindex %w[0]
    ; CHECK: [[W_0_1:%.+]] = firrtl.subfield [[W_0]](1)
    ; CHECK: [[W_0_0:%.+]] = firrtl.subfield [[W_0]](0)
    ; CHECK: firrtl.connect [[W_0_0]], [[_W_0_0]]
    ; CHECK: firrtl.attach [[W_0_1]], [[_W_0_1]]
