; RUN: firtool %s --format=fir --ir-fir 2>&1 | FileCheck %s
; RUN: firtool %s --format=fir --ir-fir --warn-on-implicit-truncation=false 2>&1 | FileCheck %s
; RUN: firtool %s --format=fir --ir-fir --warn-on-implicit-truncation --verify-diagnostics
; RUN: firtool %s --format=fir --ir-fir --warn-on-implicit-truncation=true --verify-diagnostics

FIRRTL version 4.0.0
circuit TruncWarn :
  public module TruncWarn :
    input in : UInt<7>
    output out : UInt<4>
    ; expected-warning@below {{RHS width 7 exceeds LHS width 4, inserting implicit truncation}}
    connect out, in

; CHECK: firrtl.circuit
; CHECK-NOT: RHS width 7 exceeds LHS width 4, inserting implicit truncation
