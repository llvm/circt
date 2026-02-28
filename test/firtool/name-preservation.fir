; RUN: firtool --preserve-values=named %s | FileCheck %s

FIRRTL version 4.0.0
circuit Foo:
  ; CHECK-LABEL: module Foo
  public module Foo:
    input a: {a: UInt<1>, flip b: UInt<1>}
    output b: {a: UInt<1>, flip b: UInt<1>}

    ; Unnamed wires are always removed.
    ; CHECK-NOT: wire _x_a;
    ; CHECK-NOT: wire _x_b;

    wire _x: {a: UInt<1>, flip b: UInt<1>}
    connect _x, a

    ; Preserve named wires/nodes if --preserve-values=named.
    ; CHECK:        wire x_b
    ; CHECK:        wire y
    ; CHECK:        wire x_a
    wire x: {a: UInt<1>, flip b: UInt<1>}
    connect x, _x

    ; Unnamed nodes are always removed.
    ; CHECK-NOT: wire _y_b
    node _y_b = x.b

    node y = _y_b

    connect b.a, y
    connect x.b, b.b
