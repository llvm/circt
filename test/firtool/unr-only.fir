; RUN: firtool %s --add-companion-assume --split-input-file | FileCheck %s

FIRRTL version 4.0.0
; CHECK-LABEL: module UnrOnly(
circuit UnrOnly:
  public module UnrOnly:
    input clock : Clock
    input pred : UInt<1>
    input en : UInt<1>

    ; CHECK:      `ifdef USE_UNR_ONLY_CONSTRAINTS
    ; CHECK-NEXT:   assert property (@(posedge clock) ~en | pred);
    ; CHECK-NEXT: `endif // USE_UNR_ONLY_CONSTRAINTS
    ; CHECK-NEXT: `ifdef USE_PROPERTY_AS_CONSTRAINT
    ; CHECK-NEXT:   `ifdef USE_UNR_ONLY_CONSTRAINTS
    ; CHECK-NEXT:     wire _GEN = ~en | pred;
    ; CHECK-NEXT:     always @(edge _GEN)
    ; CHECK-NEXT:       assume(_GEN);
    ; CHECK-NEXT:   `endif // USE_UNR_ONLY_CONSTRAINTS
    ; CHECK-NEXT: `endif // USE_PROPERTY_AS_CONSTRAINT
    intrinsic(circt_chisel_assert<guards="USE_UNR_ONLY_CONSTRAINTS">, clock, pred, en)

;// -----

FIRRTL version 4.0.0
; CHECK-LABEL: module UnrOnlyLabel(
circuit UnrOnlyLabel:
  public module UnrOnlyLabel:
    input clock : Clock
    input pred : UInt<1>
    input en : UInt<1>

    ; CHECK:      `ifdef USE_UNR_ONLY_CONSTRAINTS
    ; CHECK-NEXT:   assert__test: assert property (@(posedge clock) ~en | pred);
    ; CHECK-NEXT: `endif // USE_UNR_ONLY_CONSTRAINTS
    ; CHECK-NEXT: `ifdef USE_PROPERTY_AS_CONSTRAINT
    ; CHECK-NEXT:   `ifdef USE_UNR_ONLY_CONSTRAINTS
    ; CHECK-NEXT:     wire _GEN = ~en | pred;
    ; CHECK-NEXT:     always @(edge _GEN)
    ; CHECK-NEXT:       assume__test: assume(_GEN);
    ; CHECK-NEXT:   `endif // USE_UNR_ONLY_CONSTRAINTS
    ; CHECK-NEXT: `endif // USE_PROPERTY_AS_CONSTRAINT
    intrinsic(circt_chisel_assert<guards="USE_UNR_ONLY_CONSTRAINTS", label="test">, clock, pred, en)
