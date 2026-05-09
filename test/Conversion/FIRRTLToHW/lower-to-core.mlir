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

    // CHECK: [[LIT0:%.+]] = sim.fmt.literal "value="
    // CHECK: [[FMTVAL:%.+]] = sim.fmt.dec %x signed : i4
    // CHECK: [[LIT1:%.+]] = sim.fmt.literal " @ "
    // CHECK: [[HIER:%.+]] = sim.fmt.hier_path
    // CHECK: [[NL:%.+]] = sim.fmt.literal "\0A"
    // CHECK: [[MSG:%.+]] = sim.fmt.concat ([[LIT0]], [[FMTVAL]], [[LIT1]], [[HIER]], [[NL]])
    // CHECK: sim.print [[MSG]] on %clock if %enable
    // CHECK-NOT: sv.assert
    // CHECK-NOT: sv.fwrite
    firrtl.printf %clock, %enable, "value=%d @ {{}}\0A"(%x, %hier)
        : !firrtl.clock, !firrtl.uint<1>, !firrtl.sint<4>, !firrtl.fstring

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
