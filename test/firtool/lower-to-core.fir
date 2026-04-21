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
; CHECK-DAG: [[NL:%.+]] = sim.fmt.literal "\0A"
; CHECK-DAG: [[LIT:%.+]] = sim.fmt.literal "value="
; CHECK-DAG: [[CLK:%.+]] = seq.from_clock %clock
; CHECK-DAG: verif.clocked_assert %pred if %enable, posedge [[CLK]] : i1
; CHECK-DAG: [[FMT:%.+]] = sim.fmt.dec %x signed : i4
; CHECK: [[MSG:%.+]] = sim.fmt.concat ([[LIT]], [[FMT]], [[NL]])
; CHECK: sim.print [[MSG]] on %clock if %enable
; CHECK-NOT: sv.assert
; CHECK-NOT: sv.fwrite
