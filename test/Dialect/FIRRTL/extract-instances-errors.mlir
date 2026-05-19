// RUN: circt-opt --firrtl-extract-instances %s --split-input-file --verify-diagnostics

// Reject clock gate extraction on instance choice.
firrtl.circuit "ExtractClockGatesChoiceOnGate" attributes {annotations = [{class = "sifive.enterprise.firrtl.ExtractClockGatesFileAnnotation", filename = "ClockGates.txt"}]} {
  firrtl.option @Platform {
    firrtl.option_case @FPGA
  }
  firrtl.extmodule private @EICG_wrapper(in in: !firrtl.clock, in en: !firrtl.uint<1>, out out: !firrtl.clock) attributes {defname = "EICG_wrapper"}
  firrtl.extmodule private @EICG_fpga(in in: !firrtl.clock, in en: !firrtl.uint<1>, out out: !firrtl.clock) attributes {defname = "EICG_fpga"}
  firrtl.module @ExtractClockGatesChoiceOnGate(in %clock: !firrtl.clock, in %en: !firrtl.uint<1>) attributes {annotations = [{class = "sifive.enterprise.firrtl.MarkDUTAnnotation"}]} {
    // expected-error @below {{cannot extract clock gate instances through non-InstanceOp}}
    %gate_in, %gate_en, %gate_out = firrtl.instance_choice gate @EICG_wrapper alternatives @Platform {
      @FPGA -> @EICG_fpga
    } (in in: !firrtl.clock, in en: !firrtl.uint<1>, out out: !firrtl.clock)
    firrtl.connect %gate_in, %clock : !firrtl.clock, !firrtl.clock
    firrtl.connect %gate_en, %en : !firrtl.uint<1>, !firrtl.uint<1>
  }
}

// -----

// Reject memory extraction on non-instance op.
firrtl.circuit "ExtractMemsChoiceOnMem" attributes {annotations = [{class = "sifive.enterprise.firrtl.ExtractSeqMemsFileAnnotation", filename = "SeqMems.txt"}]} {
  firrtl.option @Platform {
    firrtl.option_case @FPGA
  }
  firrtl.memmodule private @mem_ext() attributes {dataWidth = 8 : ui32, depth = 8 : ui64, extraPorts = [], maskBits = 1 : ui32, numReadPorts = 0 : ui32, numReadWritePorts = 0 : ui32, numWritePorts = 0 : ui32, readLatency = 1 : ui32, writeLatency = 1 : ui32}
  firrtl.extmodule private @mem_fpga() attributes {defname = "mem_fpga"}
  firrtl.module @ExtractMemsChoiceOnMem() attributes {annotations = [{class = "sifive.enterprise.firrtl.MarkDUTAnnotation"}]} {
    // expected-error @below {{cannot extract memory instances through non-InstanceOp}}
    firrtl.instance_choice mem @mem_ext alternatives @Platform {
      @FPGA -> @mem_fpga
    } ()
  }
}

// -----

// Reject extraction through non-instance op
firrtl.circuit "ExtractClockGatesInstanceChoice" attributes {annotations = [{class = "sifive.enterprise.firrtl.ExtractClockGatesFileAnnotation", filename = "ClockGates.txt"}]} {
  firrtl.option @Platform {
    firrtl.option_case @FPGA
  }
  firrtl.extmodule private @EICG_wrapper(in in: !firrtl.clock, in en: !firrtl.uint<1>, out out: !firrtl.clock) attributes {defname = "EICG_wrapper"}
  firrtl.module private @A(in %clock: !firrtl.clock, in %en: !firrtl.uint<1>) attributes {annotations = [{class = "sifive.enterprise.firrtl.MarkDUTAnnotation"}]} {
    // expected-error @below {{cannot extract instance `gate` through a non-InstanceOp parent}}
    %gate_in, %gate_en, %gate_out = firrtl.instance gate @EICG_wrapper(in in: !firrtl.clock, in en: !firrtl.uint<1>, out out: !firrtl.clock)
    firrtl.connect %gate_in, %clock : !firrtl.clock, !firrtl.clock
    firrtl.connect %gate_en, %en : !firrtl.uint<1>, !firrtl.uint<1>
  }
  firrtl.module private @B(in %clock: !firrtl.clock, in %en: !firrtl.uint<1>) {}
  firrtl.module @ExtractClockGatesInstanceChoice(in %clock: !firrtl.clock, in %en: !firrtl.uint<1>) {
    %dut_clock, %dut_en = firrtl.instance_choice dut @A alternatives @Platform {
      @FPGA -> @B
    } (in clock: !firrtl.clock, in en: !firrtl.uint<1>)
    firrtl.connect %dut_clock, %clock : !firrtl.clock, !firrtl.clock
    firrtl.connect %dut_en, %en : !firrtl.uint<1>, !firrtl.uint<1>
  }
}
