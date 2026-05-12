; RUN: firtool --ir-hw --lower-to-core %s | FileCheck %s
FIRRTL version 4.0.0

circuit LowerToCore:
  public module LowerToCore:
    input clock: Clock
    input enable: UInt<1>
    input pred: UInt<1>
    input x: SInt<4>

    assert(clock, pred, enable, "assert failed")
    printf(clock, enable, "value=%d\n", x)

; CHECK-LABEL: hw.module @LowerToCore
; CHECK-DAG: [[CLK:%.+]] = seq.from_clock %clock
; CHECK-DAG: verif.clocked_assert %pred if %enable, posedge [[CLK]] : i1
; CHECK: hw.triggered posedge [[CLK]](%enable, %x) : i1, i4 {
; CHECK-NEXT: ^bb0(%[[EN:.*]]: i1, %[[XARG:.*]]: i4):
; CHECK: [[NL:%.+]] = sim.fmt.literal "\0A"
; CHECK: [[LIT:%.+]] = sim.fmt.literal "value="
; CHECK: [[FMT:%.+]] = sim.fmt.dec %[[XARG]] signed : i4
; CHECK: [[MSG:%.+]] = sim.fmt.concat ([[LIT]], [[FMT]], [[NL]])
; CHECK: scf.if %[[EN]] {
; CHECK-NEXT: sim.proc.print [[MSG]]
; CHECK-NOT: sim.print
; CHECK-NOT: sv.assert
; CHECK-NOT: sv.fwrite
