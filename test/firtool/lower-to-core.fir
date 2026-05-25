; RUN: firtool --ir-hw --lower-to-core %s | FileCheck %s
FIRRTL version 6.0.0

circuit LowerToCore:
  public module LowerToCore:
    input clock: Clock
    input enable: UInt<1>
    input pred: UInt<1>
    input x: SInt<4>

    assert(clock, pred, enable, "assert failed")
    printf(clock, enable, "value=%d\n", x)
    fprintf(clock, enable, "out.txt", "value=%d\n", x)
    fprintf(clock, enable, "out%d.txt", x, "value=%d\n", x)

; CHECK-LABEL: hw.module @LowerToCore
; CHECK-DAG: [[LITFILE1:%.+]] = sim.fmt.literal "out"
; CHECK-DAG: [[LITFILE2:%.+]] = sim.fmt.literal ".txt"
; CHECK-DAG: [[FMTFILE1:%.+]] = sim.fmt.literal "out.txt"
; CHECK-DAG: [[NL:%.+]] = sim.fmt.literal "\0A"
; CHECK-DAG: [[LIT:%.+]] = sim.fmt.literal "value="
; CHECK-DAG: [[CLK:%.+]] = seq.from_clock %clock
; CHECK-DAG: verif.clocked_assert %pred if %enable, posedge [[CLK]] : i1
; CHECK-DAG: [[FMT:%.+]] = sim.fmt.dec %x signed : i4
; CHECK: [[MSG:%.+]] = sim.fmt.concat ([[LIT]], [[FMT]], [[NL]])
; CHECK: sim.triggered %clock if %enable {
; CHECK-NEXT:   sim.proc.print [[MSG]]
; CHECK-NEXT: }
; CHECK: sim.triggered %clock if %enable {
; CHECK-NEXT:   [[FILE1:%.+]] = sim.get_file [[FMTFILE1]]
; CHECK-NEXT:   sim.proc.print [[MSG]] to [[FILE1]]
; CHECK-NEXT: }
; CHECK: [[FMTFILE2:%.+]] = sim.fmt.concat ([[LITFILE1]], [[FMT]], [[LITFILE2]])
; CHECK: sim.triggered %clock if %enable {
; CHECK-NEXT:   [[FILE2:%.+]] = sim.get_file [[FMTFILE2]]
; CHECK-NEXT:   sim.proc.print [[MSG]] to [[FILE2]]
; CHECK-NEXT: }
; CHECK-NOT: sv.assert
; CHECK-NOT: sv.fwrite
