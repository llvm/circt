; RUN: firtool --split-input-file %s --ir-fir | FileCheck %s
; Tests extracted from:
; - test/scala/firrtlTests/AsyncResetSpec.scala

; Complex literals should be allowed as reset values for AsyncReset.
FIRRTL version 4.0.0
circuit Foo:
  public module Foo:
    input clock : Clock
    input reset : AsyncReset
    input x : UInt<1>[4]
    output z : UInt<1>[4]
    wire literal : UInt<1>[4]
    connect literal[0], UInt<1>(0h00)
    connect literal[1], UInt<1>(0h00)
    connect literal[2], UInt<1>(0h00)
    connect literal[3], UInt<1>(0h00)
    ; CHECK: %r_0 = firrtl.regreset %clock, %reset, %c0_ui1
    ; CHECK: %r_1 = firrtl.regreset %clock, %reset, %c0_ui1
    ; CHECK: %r_2 = firrtl.regreset %clock, %reset, %c0_ui1
    ; CHECK: %r_3 = firrtl.regreset %clock, %reset, %c0_ui1
    regreset r : UInt<1>[4], clock, reset, literal
    connect r, x
    connect z, r

// -----

; Complex literals of complex literals should be allowed as reset values for
; AsyncReset.
FIRRTL version 4.0.0
circuit Foo:
  public module Foo:
    input clock : Clock
    input reset : AsyncReset
    input x : UInt<1>[4]
    output z : UInt<1>[4]
    wire literal : UInt<1>[2]
    connect literal[0], UInt<1>(0h01)
    connect literal[1], UInt<1>(0h01)
    wire complex_literal : UInt<1>[4]
    connect complex_literal[0], literal[0]
    connect complex_literal[1], literal[1]
    connect complex_literal[2], UInt<1>(0h00)
    connect complex_literal[3], UInt<1>(0h00)
    ; CHECK: %r_0 = firrtl.regreset %clock, %reset, %c1_ui1
    ; CHECK: %r_1 = firrtl.regreset %clock, %reset, %c1_ui1
    ; CHECK: %r_2 = firrtl.regreset %clock, %reset, %c0_ui1
    ; CHECK: %r_3 = firrtl.regreset %clock, %reset, %c0_ui1
    regreset r : UInt<1>[4], clock, reset, complex_literal
    connect r, x
    connect z, r

// -----

; Literals of bundle literals should be allowed as reset values for AsyncReset.
FIRRTL version 4.0.0
circuit Foo:
  public module Foo:
    input clock : Clock
    input reset : AsyncReset
    input x : UInt<1>[4]
    output z : UInt<1>[4]
    wire bundle : {a: UInt<1>, b: UInt<1>}
    connect bundle.a, UInt<1>(0h01)
    connect bundle.b, UInt<1>(0h01)
    wire complex_literal : UInt<1>[4]
    connect complex_literal[0], bundle.a
    connect complex_literal[1], bundle.b
    connect complex_literal[2], UInt<1>(0h00)
    connect complex_literal[3], UInt<1>(0h00)
    ; CHECK: %r_0 = firrtl.regreset %clock, %reset, %c1_ui1
    ; CHECK: %r_1 = firrtl.regreset %clock, %reset, %c1_ui1
    ; CHECK: %r_2 = firrtl.regreset %clock, %reset, %c0_ui1
    ; CHECK: %r_3 = firrtl.regreset %clock, %reset, %c0_ui1
    regreset r : UInt<1>[4], clock, reset, complex_literal
    connect r, x
    connect z, r

// -----

; Cast literals should be allowed as reset values for AsyncReset.
FIRRTL version 4.0.0
circuit Foo:
  public module Foo:
    input clock : Clock
    input reset : AsyncReset
    input x : SInt<4>
    output y : SInt<4>
    output z : SInt<4>
    ; CHECK: %r = firrtl.regreset %clock, %reset, %c0_si1
    regreset r : SInt<4>, clock, reset, asSInt(UInt(0))
    connect r, x
    wire w : SInt<4>
    ; CHECK: %r2 = firrtl.regreset %clock, %reset, %c-1_si4
    regreset r2 : SInt<4>, clock, reset, w
    connect r2, x
    node n = UInt(0hf)
    connect w, asSInt(n)
    connect y, r2
    connect z, r

// -----

; Unassigned asynchronously reset registers should properly constantprop.
FIRRTL version 4.0.0
circuit Foo:
  public module Foo:
    input clock : Clock
    input reset : AsyncReset
    output z : UInt<1>[4]
    wire literal : UInt<1>[2]
    connect literal[0], UInt<1>(0h01)
    connect literal[1], UInt<1>(0h01)
    wire complex_literal : UInt<1>[4]
    connect complex_literal[0], literal[0]
    connect complex_literal[1], literal[1]
    connect complex_literal[2], UInt<1>(0h00)
    connect complex_literal[3], UInt<1>(0h00)
    regreset r : UInt<1>[4], clock, reset, complex_literal
    connect z, r
    ; CHECK: firrtl.matchingconnect %z_0, %c1_ui1
    ; CHECK: firrtl.matchingconnect %z_1, %c1_ui1
    ; CHECK: firrtl.matchingconnect %z_2, %c0_ui1
    ; CHECK: firrtl.matchingconnect %z_3, %c0_ui1

// -----

; Constantly assigned asynchronously reset registers should properly
; constantprop.
FIRRTL version 4.0.0
circuit Foo:
  public module Foo:
    input clock : Clock
    input reset : AsyncReset
    output z : UInt<1>
    reg r : UInt<1>, clock
    connect r, UInt(0)
    connect z, r
    ; CHECK: firrtl.matchingconnect %z, %c0_ui1

// -----

; Constantly assigned and initialized asynchronously reset registers should
; properly constantprop.
FIRRTL version 4.0.0
circuit Foo:
  public module Foo:
    input clock : Clock
    input reset : AsyncReset
    output z : UInt<1>
    regreset r : UInt<1>, clock, reset, UInt(0)
    connect r, UInt(0)
    connect z, r
    ; CHECK: firrtl.matchingconnect %z, %c0_ui1
