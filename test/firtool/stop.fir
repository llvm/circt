; RUN: firtool %s --format=fir --ir-sv | FileCheck %s

; CHECK: sv.ifdef @STOP_COND_ {
; CHECK: } else {
; CHECK:   sv.ifdef @STOP_COND {
; CHECK:     sv.macro.def @STOP_COND_ "(`STOP_COND)"
; CHECK:   } else {
; CHECK:     sv.macro.def @STOP_COND_ "1"
; CHECK:   }
; CHECK: }

circuit StopAndFinishTest:
  module StopAndFinishTest :
    input clock : Clock
    input cond : UInt<1>

    ; CHECK: [[STOP_COND:%.+]] = sv.macro.ref @STOP_COND_() : () -> i1
    ; CHECK: [[COND:%.+]] = comb.and bin [[STOP_COND:%.+]], %cond : i1
    ; CHECK: sv.ifdef @SYNTHESIS {
    ; CHECK: } else {
    ; CHECK:   sv.always posedge %clock {
    ; CHECK:     sv.if [[COND]] {
    ; CHECK:       sv.finish 1
    ; CHECK:       sv.fatal 1
    ; CHECK:     }
    ; CHECK:   }
    ; CHECK: }

    stop(clock, cond, 0)
    stop(clock, cond, 1)
