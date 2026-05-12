// RUN: circt-opt --pass-pipeline='builtin.module(lower-firrtl-to-hw{lower-to-core=true})' %s | FileCheck %s

firrtl.circuit "LowerToCore" {
  firrtl.extmodule @AnalogSink(in a: !firrtl.analog<8>)

  // CHECK-LABEL: hw.module @LowerToCore(
  firrtl.module @LowerToCore(
      in %clock: !firrtl.clock,
      in %enable: !firrtl.uint<1>,
      in %pred: !firrtl.uint<1>,
      in %x: !firrtl.sint<4>) {
    %hier = firrtl.fstring.hierarchicalmodulename : !firrtl.fstring

    // CHECK: [[CLK:%.+]] = seq.from_clock %clock
    // CHECK: verif.clocked_assert %pred if %enable, posedge [[CLK]] : i1
    firrtl.assert %clock, %pred, %enable, "assert failed: %d"(%x)
        : !firrtl.clock, !firrtl.uint<1>, !firrtl.uint<1>, !firrtl.sint<4>
    // CHECK: [[PRINT_CLK:%.+]] = seq.from_clock %clock
    // CHECK: hw.triggered posedge [[PRINT_CLK]](%enable, %x) : i1, i4 {
    // CHECK-NEXT: ^bb0(%[[EN:.*]]: i1, %[[XARG:.*]]: i4):
    // CHECK: [[LIT0:%.+]] = sim.fmt.literal "value="
    // CHECK: [[FMTVAL:%.+]] = sim.fmt.dec %[[XARG]] signed : i4
    // CHECK: [[LIT1:%.+]] = sim.fmt.literal " @ "
    // CHECK: [[HIER:%.+]] = sim.fmt.hier_path
    // CHECK: [[NL:%.+]] = sim.fmt.literal "\0A"
    // CHECK: [[MSG:%.+]] = sim.fmt.concat ([[LIT0]], [[FMTVAL]], [[LIT1]], [[HIER]], [[NL]])
    // CHECK: scf.if %[[EN]] {
    // CHECK-NEXT: sim.proc.print [[MSG]]
    // CHECK-NOT: sim.print
    // CHECK-NOT: sv.assert
    // CHECK-NOT: sv.fwrite
    firrtl.printf %clock, %enable, "value=%d @ {{}}\0A"(%x, %hier)
        : !firrtl.clock, !firrtl.uint<1>, !firrtl.sint<4>, !firrtl.fstring

    firrtl.skip
  }

  // CHECK-LABEL: hw.module @PrintOnMemPortBackedge(
  firrtl.module @PrintOnMemPortBackedge(
      in %clock: !firrtl.clock,
      in %enable: !firrtl.uint<1>) {
    %mem = firrtl.mem Undefined {depth = 8 : i64, name = "m", portNames = ["r"], readLatency = 1 : i32, writeLatency = 1 : i32} : !firrtl.bundle<addr: uint<3>, en: uint<1>, clk: clock, data flip: uint<8>>
    %addr = firrtl.subfield %mem[addr] : !firrtl.bundle<addr: uint<3>, en: uint<1>, clk: clock, data flip: uint<8>>
    %en = firrtl.subfield %mem[en] : !firrtl.bundle<addr: uint<3>, en: uint<1>, clk: clock, data flip: uint<8>>
    %clk = firrtl.subfield %mem[clk] : !firrtl.bundle<addr: uint<3>, en: uint<1>, clk: clock, data flip: uint<8>>
    // CHECK: [[PRINT_CLK:%.+]] = seq.from_clock %clock
    // CHECK: hw.triggered posedge [[PRINT_CLK]](%enable) : i1 {
    // CHECK-NEXT: ^bb0(%[[EN:.*]]: i1):
    // CHECK: [[LIT:%.+]] = sim.fmt.literal "hit\0A"
    // CHECK: scf.if %[[EN]] {
    // CHECK-NEXT: sim.proc.print [[LIT]]
    firrtl.printf %clk, %en, "hit\0A"() : !firrtl.clock, !firrtl.uint<1>
    %c0_i3 = firrtl.constant 0 : !firrtl.uint<3>
    firrtl.connect %addr, %c0_i3 : !firrtl.uint<3>, !firrtl.uint<3>
    firrtl.connect %en, %enable : !firrtl.uint<1>, !firrtl.uint<1>
    firrtl.connect %clk, %clock : !firrtl.clock, !firrtl.clock
    firrtl.skip
  }

  // CHECK-LABEL: hw.module @MergePrintsOnResolvedSharedClock(
  firrtl.module @MergePrintsOnResolvedSharedClock(
      in %clock: !firrtl.clock,
      in %enable0: !firrtl.uint<1>,
      in %enable1: !firrtl.uint<1>) {
    %r0, %r1 = firrtl.mem Undefined {depth = 8 : i64, name = "m2", portNames = ["r0", "r1"], readLatency = 1 : i32, writeLatency = 1 : i32} : !firrtl.bundle<addr: uint<3>, en: uint<1>, clk: clock, data flip: uint<8>>, !firrtl.bundle<addr: uint<3>, en: uint<1>, clk: clock, data flip: uint<8>>
    %r0_addr = firrtl.subfield %r0[addr] : !firrtl.bundle<addr: uint<3>, en: uint<1>, clk: clock, data flip: uint<8>>
    %r0_en = firrtl.subfield %r0[en] : !firrtl.bundle<addr: uint<3>, en: uint<1>, clk: clock, data flip: uint<8>>
    %r0_clk = firrtl.subfield %r0[clk] : !firrtl.bundle<addr: uint<3>, en: uint<1>, clk: clock, data flip: uint<8>>
    %r1_addr = firrtl.subfield %r1[addr] : !firrtl.bundle<addr: uint<3>, en: uint<1>, clk: clock, data flip: uint<8>>
    %r1_en = firrtl.subfield %r1[en] : !firrtl.bundle<addr: uint<3>, en: uint<1>, clk: clock, data flip: uint<8>>
    %r1_clk = firrtl.subfield %r1[clk] : !firrtl.bundle<addr: uint<3>, en: uint<1>, clk: clock, data flip: uint<8>>
    // CHECK: [[MERGED_CLK:%.+]] = seq.from_clock %clock
    // CHECK-COUNT-1: hw.triggered posedge [[MERGED_CLK]](%enable0, %enable1) : i1, i1 {
    // CHECK-NEXT: ^bb0(%[[EN0:.*]]: i1, %[[EN1:.*]]: i1):
    // CHECK: [[LITA:%.+]] = sim.fmt.literal "A\0A"
    // CHECK: [[LITB:%.+]] = sim.fmt.literal "B\0A"
    // CHECK: scf.if %[[EN0]] {
    // CHECK-NEXT: sim.proc.print [[LITA]]
    // CHECK: scf.if %[[EN1]] {
    // CHECK-NEXT: sim.proc.print [[LITB]]
    firrtl.printf %r0_clk, %r0_en, "A\0A"() : !firrtl.clock, !firrtl.uint<1>
    firrtl.printf %r1_clk, %r1_en, "B\0A"() : !firrtl.clock, !firrtl.uint<1>
    %c0_i3 = firrtl.constant 0 : !firrtl.uint<3>
    firrtl.connect %r0_addr, %c0_i3 : !firrtl.uint<3>, !firrtl.uint<3>
    firrtl.connect %r0_en, %enable0 : !firrtl.uint<1>, !firrtl.uint<1>
    firrtl.connect %r0_clk, %clock : !firrtl.clock, !firrtl.clock
    firrtl.connect %r1_addr, %c0_i3 : !firrtl.uint<3>, !firrtl.uint<3>
    firrtl.connect %r1_en, %enable1 : !firrtl.uint<1>, !firrtl.uint<1>
    firrtl.connect %r1_clk, %clock : !firrtl.clock, !firrtl.clock
    firrtl.skip
  }

  // CHECK-LABEL: hw.module @AttachToPort(inout %a : i8)
  // CHECK-NEXT: hw.instance "sink" @AnalogSink(a: %a: !hw.inout<i8>) -> ()
  // CHECK-NEXT: hw.output
  // CHECK-NOT: sv.
  firrtl.module @AttachToPort(out %a: !firrtl.analog<8>) {
    %sink = firrtl.instance sink @AnalogSink(in a: !firrtl.analog<8>)
    firrtl.attach %a, %sink : !firrtl.analog<8>, !firrtl.analog<8>
  }
}
