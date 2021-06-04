; RUN: circt-translate --import-firrtl %s | circt-opt --convert-firrtl-to-llhd | FileCheck %s

circuit test_mod :
  module test_mod :
    input a: UInt<1>
    output b: UInt<1>
    b <= a

; CHECK-LABEL: llhd.entity @test_mod (
; CHECK-SAME: %[[PORT_A:.*]] : !llhd.sig<i1>) -> (
; CHECK-SAME: %[[PORT_B:.*]] : !llhd.sig<i1>) {
; CHECK: %[[PRB:.*]] = llhd.prb %[[PORT_A]] : !llhd.sig<i1>
; CHECK: %[[DELTA:.*]] = llhd.const #llhd.time<0s, 1d, 0e> : !llhd.time
; CHECK: %[[TRUE:.*]] = llhd.const true : i1
; CHECK: llhd.drv %[[PORT_B]], %[[PRB:.*]] after %[[DELTA]] if %[[TRUE]] : !llhd.sig<i1>
; CHECK-NEXT: }
