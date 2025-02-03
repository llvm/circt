; RUN: firtool %s --format=fir --ir-sv | FileCheck %s

FIRRTL version 4.0.0
circuit PrintTest:
  ; CHECK-LABEL: @PrintTest
  public module PrintTest :
    input clock : Clock
    input cond : UInt<1>
    input var : UInt<32>
    printf(clock, cond, "test %d\n", var)

    ; CHECK:      sv.ifdef  @SYNTHESIS {
    ; CHECK-NEXT: } else {
    ; CHECK-NEXT:   sv.always posedge %clock {
    ; CHECK-NEXT:     [[PRINTF_COND:%.+]] = sv.macro.ref.expr @PRINTF_COND_() : () -> i1
    ; CHECK-NEXT:     [[COND:%.+]] = comb.and [[PRINTF_COND]], %cond : i1
    ; CHECK-NEXT:     sv.if [[COND]] {
    ; CHECK-NEXT:       [[PRINTF_FD:%.+]] = sv.macro.ref.expr @PRINTF_FD_() : () -> i32
    ; CHECK-NEXT:       sv.fwrite [[PRINTF_FD]], "test %d\0A"(%var) : i32
    ; CHECK-NEXT:     }
    ; CHECK-NEXT:   }
    ; CHECK-NEXT: }
