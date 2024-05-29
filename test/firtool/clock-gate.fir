; RUN: firtool %s | FileCheck %s --check-prefixes=CHECK,CHECK-DEFAULT
; RUN: firtool %s --ckg-name=MyClockGate --ckg-input=CI --ckg-output=CO --ckg-enable=E --ckg-test-enable=TE | FileCheck %s --check-prefixes=CHECK,CHECK-CUSTOM
FIRRTL version 4.0.0

circuit Foo:
  public module Foo:
    input clk: Clock
    input enable: UInt<1>
    output gated_clk: Clock

    connect gated_clk, intrinsic(circt_clock_gate : Clock, clk, enable)

; CHECK-LABEL: module Foo(

; CHECK-DEFAULT: EICG_wrapper ckg (
; CHECK-DEFAULT: .in (clk)
; CHECK-DEFAULT: .out (gated_clk)
; CHECK-DEFAULT: .en (enable)
; CHECK-DEFAULT: .test_en (1'h0)

; CHECK-CUSTOM: MyClockGate ckg (
; CHECK-CUSTOM: .CI (clk)
; CHECK-CUSTOM: .CO (gated_clk)
; CHECK-CUSTOM: .E (enable)
; CHECK-CUSTOM: .TE (1'h0)
