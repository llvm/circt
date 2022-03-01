; RUN: firtool %s | FileCheck %s --check-prefixes=CHECK,NAMES
; RUN: firtool %s -disable-name-preservation | FileCheck %s --check-prefixes=CHECK,NO_NAMES

circuit Foo:
  ; CHECK-LABEL: module Foo
  module Foo:
    input a: {a: UInt<1>, flip b: UInt<1>}
    output b: {a: UInt<1>, flip b: UInt<1>}

    ; Unnamed wires are always removed.
    ; CHECK-NOT: wire _x_a;
    ; CHECK-NOT: wire _x_b;

    wire _x: {a: UInt<1>, flip b: UInt<1>}
    _x <= a

    ; Default behavior is to preserve named wires.
    ; NAMES:        wire x_a;
    ; NAMES:        wire x_b;
    ; With -disable-name-preservation, named wires are removed.
    ; NO_NAMES-NOT: wire x_b;
    ; NO_NAMES-NOT: wire x_a;
    wire x: {a: UInt<1>, flip b: UInt<1>}
    x <= _x

    ; Unnamed nodes are always removed.
    ; CHECK-NOT: wire _y_a;
    node _y_a = x.a

    ; Default behavior is to preserve named nodes.
    ; NAMES:        wire y;
    ; With -disable-name-preservation, named nodes are removed.
    ; NO-NAMES-NOT: wire y;
    node y = _y_a

    b.a <= y
    x.b <= b.b
