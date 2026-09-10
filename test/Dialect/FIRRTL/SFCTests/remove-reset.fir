; RUN: firtool --split-input-file %s --ir-fir | FileCheck %s
; Tests extracted from:
; - test/scala/firrtlTests/transforms/RemoveResetSpec.scala

; Should not generate a reset mux for an invalid init, given a 1-bit register
; 'foo' initialized to invalid, 1-bit wire 'bar'.
FIRRTL version 4.0.0
circuit Example :
  public module Example :
    input clock : Clock
    input arst : AsyncReset
    input srst : UInt<1>
    input in : UInt<1>
    output out : UInt<1>[2]
    wire bar : UInt<1>
    invalidate bar
    ; CHECK: %foo0 = firrtl.reg %clock
    ; CHECK: %foo1 = firrtl.reg %clock
    regreset foo0 : UInt<1>, clock, arst, bar
    regreset foo1 : UInt<1>, clock, srst, bar
    connect foo0, in
    connect foo1, in
    connect out[0], foo0
    connect out[1], foo1

// -----

; Should generate a reset mux for only the portion of an invalid aggregate that
; is reset, given aggregate register 'foo' with 2-bit field 'a' and 1-bit field
; 'b', and aggregate, invalid wire 'bar' with the same fields, and 'foo' is
; initialized to 'bar', and 'bar.a[1]' connected to zero.
FIRRTL version 4.0.0
circuit Example1 :
  public module Example1 :
    input clock : Clock
    input arst : AsyncReset
    input srst : UInt<1>
    input in :  {a : UInt<1>[2], b : UInt<1>}
    output out :  {a : UInt<1>[2], b : UInt<1>}[2]

    wire bar : {a : UInt<1>[2], b : UInt<1>}
    invalidate bar
    connect bar.a[1], UInt<1>(0)

    ; CHECK: %foo0_a_0 = firrtl.reg %clock
    ; CHECK: %foo0_a_1 = firrtl.regreset %clock, %arst,
    ; CHECK: %foo0_b = firrtl.reg %clock
    ; CHECK: %foo1_a_0 = firrtl.reg %clock
    ; CHECK: %foo1_a_1 = firrtl.regreset %clock, %srst,
    ; CHECK: %foo1_b = firrtl.reg %clock
    regreset foo0 : {a : UInt<1>[2], b : UInt<1>}, clock, arst, bar
    regreset foo1 : {a : UInt<1>[2], b : UInt<1>}, clock, srst, bar
    connect foo0, in
    connect foo1, in
    connect out[0], foo0
    connect out[1], foo1

// -----

; Should propagate invalidations across connects, given aggregate register 'foo'
; with 1-bit field 'a' and 1-bit field 'b', and aggregate, invalid wires 'bar'
; and 'baz' with the same fields, and 'foo' is initialized to 'baz', and 'bar.a'
; is connected to zero, and 'baz' is connected to 'bar'.
FIRRTL version 4.0.0
circuit Example2 :
  public module Example2 :
    input clock : Clock
    input arst : AsyncReset
    input srst : UInt<1>
    input in : { a : UInt<1>, b : UInt<1> }
    output out : { a : UInt<1>, b : UInt<1> }[2]

    wire bar : { a : UInt<1>, b : UInt<1> }
    invalidate bar
    connect bar.a, UInt<1>(0)

    wire baz : { a : UInt<1>, b : UInt<1> }
    invalidate baz
    connect baz, bar

    ; CHECK: %foo0_a = firrtl.regreset %clock, %arst,
    ; CHECK: %foo0_b = firrtl.reg %clock
    ; CHECK: %foo1_a = firrtl.regreset %clock, %srst,
    ; CHECK: %foo1_b = firrtl.reg %clock
    regreset foo0 : { a : UInt<1>, b : UInt<1> }, clock, arst, baz
    regreset foo1 : { a : UInt<1>, b : UInt<1> }, clock, srst, baz
    connect foo0, in
    connect foo1, in
    connect out[0], foo0
    connect out[1], foo1

// -----

; Should convert a reset wired to UInt<0> to a canonical non-reset, given foo's
; reset is connected to zero.
FIRRTL version 4.0.0
circuit Example3 :
  public module Example3 :
    input clock : Clock
    input in : UInt<2>
    output out : UInt<2>[3]

    wire rst : Reset
    wire arst : AsyncReset
    wire srst : UInt<1>

    connect rst, UInt<1>(0)
    connect arst, asAsyncReset(UInt<1>(0))
    connect srst, UInt<1>(0)

    ; CHECK: %foo0 = firrtl.reg %clock
    ; CHECK: %foo1 = firrtl.reg %clock
    ; CHECK: %foo2 = firrtl.reg %clock
    regreset foo0 : UInt<2>, clock, rst, UInt(3)
    regreset foo1 : UInt<2>, clock, arst, UInt(3)
    regreset foo2 : UInt<2>, clock, srst, UInt(3)
    connect foo0, in
    connect foo1, in
    connect foo2, in
    connect out[0], foo0
    connect out[1], foo1
    connect out[2], foo2
