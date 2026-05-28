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
    // CHECK: [[MSG0:%.+]] = sim.fmt.concat ([[LIT0]], [[FMTVAL]], [[LIT1]], [[HIER]], [[NL]])
    // CHECK: [[STDERR0:%.+]] = sim.stderr_stream
    // CHECK-NOT: sv.assert
    // CHECK-NOT: sv.fwrite
    firrtl.printf %clock, %enable, "value=%d @ {{}}\0A"(%x, %hier)
        : !firrtl.clock, !firrtl.uint<1>, !firrtl.sint<4>, !firrtl.fstring

    // CHECK: [[FMTFILE1:%.+]] = sim.fmt.literal "out.txt"
    // CHECK: [[MSG1:%.+]] = sim.fmt.concat
    firrtl.fprintf %clock, %enable, "out.txt"(), "value=%d @ {{}}\0A"(%x, %hier)
        : !firrtl.clock, !firrtl.uint<1>, !firrtl.sint<4>, !firrtl.fstring
    
    // CHECK: [[LIT2:%.+]] = sim.fmt.literal "out"
    // CHECK: [[FILEVAL:%.+]] = sim.fmt.dec %x signed : i4
    // CHECK: [[LIT3:%.+]] = sim.fmt.literal ".txt"
    // CHECK: [[FMTFILE2:%.+]] = sim.fmt.concat ([[LIT2]], [[FILEVAL]], [[LIT3]])
    // CHECK: [[MSG2:%.+]] = sim.fmt.concat
    firrtl.fprintf %clock, %enable, "out%d.txt"(%x), "value=%d @ {{}}\0A"(%x, %hier)
        : !firrtl.clock, !firrtl.uint<1>, !firrtl.sint<4>, !firrtl.sint<4>, !firrtl.fstring

    // CHECK: [[LIT4:%.+]] = sim.fmt.literal "out"
    // CHECK: [[FILEVAL:%.+]] = sim.fmt.dec %x signed : i4
    // CHECK: [[LIT5:%.+]] = sim.fmt.literal ".txt
    // CHECK: [[FMTFILE3:%.+]] = sim.fmt.concat ([[LIT4]], [[FILEVAL]], [[LIT5]])
    firrtl.fflush %clock, %enable, "out%d.txt"(%x)
        : !firrtl.clock, !firrtl.uint<1>, !firrtl.sint<4>

    // CHECK: [[STDERR1:%.+]] = sim.stderr_stream
    // CHECK: sim.triggered %clock {
    // CHECK-NEXT:   scf.if %enable {
    // CHECK-NEXT:     sim.proc.print [[MSG0]] to [[STDERR0]]
    // CHECK-NEXT:     [[FILE1:%.+]] = sim.get_file [[FMTFILE1]]
    // CHECK-NEXT:     sim.proc.print [[MSG1]] to [[FILE1]]
    // CHECK-NEXT:     [[FILE2:%.+]] = sim.get_file [[FMTFILE2]]
    // CHECK-NEXT:     sim.proc.print [[MSG2]] to [[FILE2]]
    // CHECK-NEXT:     [[FILE3:%.+]] = sim.get_file [[FMTFILE3]]
    // CHECK-NEXT:     sim.flush [[FILE3]]
    // CHECK-NEXT:     sim.flush [[STDERR1]]
    // CHECK-NEXT:   }
    // CHECK-NEXT: }
    firrtl.fflush %clock, %enable
        : !firrtl.clock, !firrtl.uint<1>

    // CHECK: [[TIME:%.+]] = sim.fmt.current_time
    // CHECK: [[LIT4:%.+]] = sim.fmt.literal "\0A"
    // CHECK: [[MSG:%.+]] = sim.fmt.concat ([[TIME]], [[LIT4]])
    // CHECK: [[STDERR:%.+]] = sim.stderr_stream
    // CHECK: sim.triggered %clock if %enable {
    // CHECK-NEXT:   sim.proc.print [[MSG]] to [[STDERR]]
    // CHECK-NEXT: }
    %time = firrtl.fstring.time : !firrtl.fstring
    firrtl.printf %clock, %enable, "{{}}\0A"(%time)
        : !firrtl.clock, !firrtl.uint<1>, !firrtl.fstring

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

  // CHECK-LABEL: hw.module @InterleavedConditions(
  firrtl.module @InterleavedConditions(
      in %clock: !firrtl.clock,
      in %enable_a: !firrtl.uint<1>,
      in %enable_b: !firrtl.uint<1>) {
    // CHECK: [[A0:%.+]] = sim.fmt.literal "a0\0A"
    // CHECK: [[STDERR0:%.+]] = sim.stderr_stream
    // CHECK: [[B0:%.+]] = sim.fmt.literal "b0\0A"
    // CHECK: [[STDERR1:%.+]] = sim.stderr_stream
    // CHECK: [[A1:%.+]] = sim.fmt.literal "a1\0A"
    // CHECK: [[STDERR2:%.+]] = sim.stderr_stream
    // CHECK: sim.triggered %clock {
    // CHECK-NEXT:   scf.if %enable_a {
    // CHECK-NEXT:     sim.proc.print [[A0]] to [[STDERR0]]
    // CHECK-NEXT:   }
    // CHECK-NEXT:   scf.if %enable_b {
    // CHECK-NEXT:     sim.proc.print [[B0]] to [[STDERR1]]
    // CHECK-NEXT:   }
    // CHECK-NEXT:   scf.if %enable_a {
    // CHECK-NEXT:     sim.proc.print [[A1]] to [[STDERR2]]
    // CHECK-NEXT:   }
    // CHECK-NEXT: }
    firrtl.printf %clock, %enable_a, "a0\0A"()
        : !firrtl.clock, !firrtl.uint<1>
    firrtl.printf %clock, %enable_b, "b0\0A"()
        : !firrtl.clock, !firrtl.uint<1>
    firrtl.printf %clock, %enable_a, "a1\0A"()
        : !firrtl.clock, !firrtl.uint<1>
    firrtl.skip
  }
}
