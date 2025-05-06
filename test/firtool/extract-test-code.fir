; RUN: firtool %s -extract-test-code | FileCheck %s
; RUN: firtool %s -extract-test-code -etc-disable-instance-extraction | FileCheck %s --check-prefix=DISABLEINSTANCE
; RUN: firtool %s -extract-test-code -etc-disable-module-inlining | FileCheck %s --check-prefix=DISABLEMODULE

FIRRTL version 4.0.0
circuit Top:
  module Foo:
    input a : UInt<1>
    output b : UInt<1>
    connect b, a

  ; Ensure foo is extracted by default.
  ; CHECK-LABEL: module InstanceExtracted_assert(
  ; CHECK: Foo foo

  ; Ensure foo is not extracted when disabled.
  ; DISABLEINSTANCE-LABEL: module InstanceExtracted(
  ; DISABLEINSTANCE: Foo foo

  module InstanceExtracted:
    input clock : Clock
    input cond : UInt<1>
    output out : UInt<1>

    wire b : UInt<1>
    inst foo of Foo
    connect foo.a, cond
    connect b, foo.b

    assert(clock, cond, b, "Some assertion")

    connect out, cond

  ; Ensure InputOnly is inlined by default.
  ; CHECK-NOT: module InputOnly(

  ; Ensure InputOnly is not inlined when disabled.
  ; DISABLEMODULE-LABEL: module InputOnly(

  module InputOnly:
    input clock : Clock
    input cond : UInt<1>
    input en : UInt<1>
    assert(clock, cond, en, "Some assertion")

  ; CHECK: module Top_assert(
  ; CHECK-NOT: endmodule
  ; CHECK:   wire _GEN = ~en; 
  ; CHECK:   foo: assert property (disable iff (_GEN) cond);
  ; CHECK: endmodule

  public module Top:
    input clock : Clock
    input cond  : UInt<1>
    input en    : UInt<1>
    output out  : UInt<1>

    inst instance_extracted of InstanceExtracted
    connect instance_extracted.clock, clock
    connect instance_extracted.cond, cond
    connect out, instance_extracted.out

    inst input_only of InputOnly
    connect input_only.clock, clock
    connect input_only.cond, cond
    connect input_only.en, en

    intrinsic(circt_verif_assert<label="foo">, cond, en)
