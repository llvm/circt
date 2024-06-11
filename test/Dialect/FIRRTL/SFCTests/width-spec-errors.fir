; RUN: firtool --split-input-file --verify-diagnostics %s
; Tests extracted from:
; - test/scala/firrtlTests/WidthSpec.scala

; Dshl by more than 31 bits should result in an error
FIRRTL version 4.0.0
circuit Unit :
  public module Unit :
    input x : UInt<3>
    input y : UInt<32>
    ; expected-error @+1 {{shift amount too large: second operand of dshl is wider than 31 bits}}
    node z = dshl(x, y)

// -----

; Dshl by to more than 31 bits total width should result in an error
FIRRTL version 4.0.0
circuit Unit :
  public module Unit :
    input x : UInt<1073741825>
    input y : UInt<30>
    ; expected-error @+1 {{shift amount too large: first operand shifted by maximum amount exceeds maximum width}}
    node z = dshl(x, y)

// -----

; Casting a multi-bit signal to Clock should result in error
FIRRTL version 4.0.0
circuit Unit :
  public module Unit :
    input i: UInt<2>
    ; expected-error @+1 {{must be 1-bit uint/sint/analog, reset, asyncreset, or clock}}
    node x = asClock(i)

// -----

; Casting a multi-bit signal to AsyncReset should result in error
FIRRTL version 4.0.0
circuit Unit :
  public module Unit :
    input i: UInt<2>
    ; expected-error @+1 {{operand must be single bit scalar type}}
    node x = asAsyncReset(i)

// -----

; Width >= MaxWidth should result in an error
FIRRTL version 4.0.0
circuit Unit :
  public module Unit :
    ; expected-error @+1 {{value is too big to handle}}
    input x: UInt<2147483648>

// -----

; Circular reg depending on reg + 1 should error
FIRRTL version 4.0.0
circuit Unit :
  public module Unit :
    input clock: Clock
    input reset: UInt<1>
    ; expected-error @+1 {{'firrtl.regreset' op is constrained to be wider than itself}}
    reg r : UInt, clock with :
      reset => (reset, UInt(3))
    ; expected-note @+1 {{constrained width W >= W+1 here:}}
    node T_7 = add(r, r)
    ; expected-note @+1 {{constrained width W >= W+1 here:}}
    r <= T_7

// -----

; Add of UInt<2> and SInt<2> should error
FIRRTL version 4.0.0
circuit Unit :
  public module Unit :
    input x: UInt<2>
    input y: SInt<2>
    ; expected-error @+1 {{operand signedness must match}}
    node z = add(x, y)

// -----

; SInt<2> - UInt<3> should error
FIRRTL version 4.0.0
circuit Unit :
  public module Unit :
    input x: UInt<3>
    input y: SInt<2>
    ; expected-error @+1 {{operand signedness must match}}
    node z = sub(y, x)

// -----

; Should provide a good error message with a full target if a user forgets an
; assign.
FIRRTL version 4.0.0
circuit Foo :
  public module Foo :
    input clock : Clock
    inst bar of Bar
  module Bar :
    ; expected-error @+1 {{uninferred width: wire "a.c.e" is unconstrained}}
    wire a: { b : UInt<1>, c : { d : UInt<1>, e : UInt } }
    a is invalid
