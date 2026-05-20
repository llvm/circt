// RUN: circt-opt --firrtl-extract-instances %s --split-input-file --verify-diagnostics

// Reject extraction on instance choice.
firrtl.circuit "ExtractMemsChoiceOnMem" attributes {annotations = [{class = "sifive.enterprise.firrtl.ExtractSeqMemsFileAnnotation", filename = "SeqMems.txt"},
                                                                   {class = "sifive.enterprise.firrtl.ExtractClockGatesFileAnnotation", filename = "ClockGates.txt"}]} {
  firrtl.option @Platform {
    firrtl.option_case @FPGA
  }
  firrtl.memmodule private @mem_ext() attributes {dataWidth = 8 : ui32, depth = 8 : ui64, extraPorts = [], maskBits = 1 : ui32, numReadPorts = 0 : ui32, numReadWritePorts = 0 : ui32, numWritePorts = 0 : ui32, readLatency = 1 : ui32, writeLatency = 1 : ui32}
  firrtl.extmodule private @mem_fpga() attributes {defname = "mem_fpga"}
  firrtl.extmodule private @EICG_wrapper() attributes {defname = "EICG_wrapper"}

  firrtl.module @ExtractMemsChoiceOnMem() attributes {annotations = [{class = "sifive.enterprise.firrtl.MarkDUTAnnotation"}]} {
    // expected-error @below {{cannot extract memory instances through non-InstanceOp}}
    firrtl.instance_choice mem @mem_ext alternatives @Platform {
      @FPGA -> @mem_fpga
    } ()

    // expected-error @below {{cannot extract clock gate instances through non-InstanceOp}}
    firrtl.instance_choice gate @EICG_wrapper alternatives @Platform { @FPGA -> @EICG_wrapper } ()
  }
}

// -----

// Reject extraction through non-instance op
firrtl.circuit "ExtractClockGatesInstanceChoice" attributes {annotations = [{class = "sifive.enterprise.firrtl.ExtractClockGatesFileAnnotation", filename = "ClockGates.txt"}]} {
  firrtl.option @Platform {
    firrtl.option_case @FPGA
  }
  firrtl.extmodule private @EICG_wrapper() attributes {defname = "EICG_wrapper"}
  firrtl.module private @A() attributes {annotations = [{class = "sifive.enterprise.firrtl.MarkDUTAnnotation"}]} {
    // expected-error @below {{cannot extract instance `gate` through a non-InstanceOp parent}}
    firrtl.instance gate @EICG_wrapper()
  }
  firrtl.module private @B() {}
  firrtl.module @ExtractClockGatesInstanceChoice() {
    firrtl.instance_choice dut @A alternatives @Platform {@FPGA -> @B} ()
  }
}
