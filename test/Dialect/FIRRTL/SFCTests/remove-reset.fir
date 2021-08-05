; RUN: firtool --split-input-file %s | FileCheck %s
; Tests extracted from:
; - test/scala/firrtlTests/transforms/RemoveResetSpec.scala


; Should not generate a reset mux for an invalid init, given a 1-bit register
; 'foo' initialized to invalid, 1-bit wire 'bar'.
circuit Example :
  module Example :
    input clock : Clock
    input rst : Reset
    input arst : AsyncReset
    input srst : UInt<1>
    input in : UInt<1>
    output out : UInt<1>[3]
    wire bar : UInt<1>
    bar is invalid
    ; CHECK: %foo0 = firrtl.reg %clock :
    ; CHECK: %foo1 = firrtl.reg %clock :
    ; CHECK: %foo2 = firrtl.reg %clock :
    reg foo0 : UInt<1>, clock with : (reset => (rst, bar))
    reg foo1 : UInt<1>, clock with : (reset => (arst, bar))
    reg foo2 : UInt<1>, clock with : (reset => (srst, bar))
    foo0 <= in
    foo1 <= in
    foo2 <= in
    out[0] <= foo0
    out[1] <= foo1
    out[2] <= foo2

// -----

; Should generate a reset mux for only the portion of an invalid aggregate that
; is reset, given aggregate register 'foo' with 2-bit field 'a' and 1-bit field
; 'b', and aggregate, invalid wire 'bar' with the same fields, and 'foo' is
; initialized to 'bar', and 'bar.a[1]' connected to zero.
circuit Example :
  module Example :
    input clock : Clock
    input rst : Reset
    input arst : AsyncReset
    input srst : UInt<1>
    input in :  {a : UInt<1>[2], b : UInt<1>}
    output out :  {a : UInt<1>[2], b : UInt<1>}[3]

    wire bar : {a : UInt<1>[2], b : UInt<1>}
    bar is invalid
    bar.a[1] <= UInt<1>(0)

    ; CHECK: %foo0_a_0 = firrtl.reg %clock :
    ; CHECK: %foo0_a_1 = firrtl.regreset %clock, %rst,
    ; CHECK: %foo0_b = firrtl.reg %clock :
    ; CHECK: %foo1_a_0 = firrtl.reg %clock :
    ; CHECK: %foo1_a_1 = firrtl.regreset %clock, %arst,
    ; CHECK: %foo1_b = firrtl.reg %clock :
    ; CHECK: %foo2_a_0 = firrtl.reg %clock :
    ; CHECK: %foo2_a_1 = firrtl.regreset %clock, %srst,
    ; CHECK: %foo2_b = firrtl.reg %clock :
    reg foo0 : {a : UInt<1>[2], b : UInt<1>}, clock with : (reset => (rst, bar))
    reg foo1 : {a : UInt<1>[2], b : UInt<1>}, clock with : (reset => (arst, bar))
    reg foo2 : {a : UInt<1>[2], b : UInt<1>}, clock with : (reset => (srst, bar))
    foo0 <= in
    foo1 <= in
    foo2 <= in
    out[0] <= foo0
    out[1] <= foo1
    out[2] <= foo2

// -----

; Should propagate invalidations across connects, given aggregate register 'foo'
; with 1-bit field 'a' and 1-bit field 'b', and aggregate, invalid wires 'bar'
; and 'baz' with the same fields, and 'foo' is initialized to 'baz', and 'bar.a'
; is connected to zero, and 'baz' is connected to 'bar'.
circuit Example :
  module Example :
    input clock : Clock
    input rst : Reset
    input arst : AsyncReset
    input srst : UInt<1>
    input in : { a : UInt<1>, b : UInt<1> }
    output out : { a : UInt<1>, b : UInt<1> }[3]

    wire bar : { a : UInt<1>, b : UInt<1> }
    bar is invalid
    bar.a <= UInt<1>(0)

    wire baz : { a : UInt<1>, b : UInt<1> }
    baz is invalid
    baz <= bar

    ; CHECK: %foo0_a = firrtl.regreset %clock, %rst,
    ; CHECK: %foo0_b = firrtl.reg %clock :
    ; CHECK: %foo1_a = firrtl.regreset %clock, %arst,
    ; CHECK: %foo1_b = firrtl.reg %clock :
    ; CHECK: %foo2_a = firrtl.regreset %clock, %srst,
    ; CHECK: %foo2_b = firrtl.reg %clock :
    reg foo0 : { a : UInt<1>, b : UInt<1> }, clock with : (reset => (rst, baz))
    reg foo1 : { a : UInt<1>, b : UInt<1> }, clock with : (reset => (arst, baz))
    reg foo2 : { a : UInt<1>, b : UInt<1> }, clock with : (reset => (srst, baz))
    foo0 <= in
    foo1 <= in
    foo2 <= in
    out[0] <= foo0
    out[1] <= foo1
    out[2] <= foo2

// -----

; Should convert a reset wired to UInt<0> to a canonical non-reset, given foo's
; reset is connected to zero.
circuit Example :
  module Example :
    input clock : Clock
    input in : UInt<2>
    output out : UInt<2>[3]

    wire rst : Reset
    wire arst : AsyncReset
    wire srst : UInt<1>

    rst <= UInt<1>(0)
    arst <= asAsyncReset(UInt<1>(0))
    srst <= UInt<1>(0)

    ; CHECK: %foo0 = firrtl.reg %clock :
    ; CHECK: %foo1 = firrtl.reg %clock :
    ; CHECK: %foo2 = firrtl.reg %clock :
    reg foo0 : UInt<2>, clock with : (reset => (rst, UInt(3)))
    reg foo1 : UInt<2>, clock with : (reset => (arst, UInt(3)))
    reg foo2 : UInt<2>, clock with : (reset => (srst, UInt(3)))
    foo0 <= in
    foo1 <= in
    foo2 <= in
    out[0] <= foo0
    out[1] <= foo1
    out[2] <= foo2
