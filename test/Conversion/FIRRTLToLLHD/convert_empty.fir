; RUN: firtool %s --format=fir -mlir | circt-opt --convert-firrtl-to-llhd | FileCheck %s

circuit test_mod :
  module test_mod :
    input a: UInt<1>
    output b: UInt<1>
    b <= a

; CHECK-LABEL: llhd.entity @test_mod (
; CHECK-SAME: %[[PORT_A:.*]]: !llhd.sig<i1>) -> (
; CHECK-SAME: %[[PORT_B:.*]]: !llhd.sig<i1>) {
; CHECK: %[[VAL_0:.*]] = llhd.prb %[[PORT_A]] : !llhd.sig<i1>
; CHECK: %[[TIME:.*]] = #llhd.time<0ns, 1d, 0e>
; CHECK: llhd.drv %[[PORT_B]], %[[VAL_0]] after %[[TIME]] : !llhd.sig<i1>
; CHECK: con %[[PORT_B]], %[[PORT_A]] : (!llhd.sig<i1>, !llhd.sig<i1>)
; CHECK: }
