// RUN: circt-opt -firrtl-add-seqmem-ports -verify-diagnostics -split-input-file %s

// expected-error@below {{MetadataDirAnnotation requires field 'dirname' of string type}}
firrtl.circuit "Simple" attributes {
  annotations = [
    {
      class = "sifive.enterprise.firrtl.MetadataDirAnnotation"
    }
  ]
} {
  firrtl.module @Simple() {}
}

// -----

// expected-error@below {{AddSeqMemPortsFileAnnotation requires field 'filename' of string type}}
firrtl.circuit "Simple" attributes {
  annotations = [
    {
      class = "sifive.enterprise.firrtl.AddSeqMemPortsFileAnnotation"
    }
  ]
} {
  firrtl.module @Simple() {}
}

// -----

// expected-error@below {{circuit has two AddSeqMemPortsFileAnnotation annotations}}
firrtl.circuit "Simple" attributes {
  annotations = [
    {
      class = "sifive.enterprise.firrtl.AddSeqMemPortsFileAnnotation",
      filename = "test"
    },
    {
      class = "sifive.enterprise.firrtl.AddSeqMemPortsFileAnnotation",
      filename = "test"
    }
  ]
} {
  firrtl.module @Simple() {}
}

// -----

// expected-error@below {{AddSeqMemPortAnnotation requires field 'name' of string type}}
firrtl.circuit "Simple" attributes {
  annotations = [
    {
      class = "sifive.enterprise.firrtl.AddSeqMemPortAnnotation",
      input = true,
      width = 5
    }
  ]
} {
  firrtl.module @Simple() { }
}

// -----

// expected-error@below {{AddSeqMemPortAnnotation requires field 'input' of boolean type}}
firrtl.circuit "Simple" attributes {
  annotations = [
    {
      class = "sifive.enterprise.firrtl.AddSeqMemPortAnnotation",
      name = "user_input",
      width = 5
    }
  ]
} {
  firrtl.module @Simple() { }
}

// -----

// expected-error@below {{AddSeqMemPortAnnotation requires field 'width' of integer type}}
firrtl.circuit "Simple" attributes {
  annotations = [
    {
      class = "sifive.enterprise.firrtl.AddSeqMemPortAnnotation",
      name = "user_input",
      input = true
    }
  ]
} {
  firrtl.module @Simple() { }
}

// -----

// This has a memory that is instantiated under the design-under-test _and_
// under a layer block.
firrtl.circuit "LayerBlock" attributes {
  annotations = [
    {
      class = "sifive.enterprise.firrtl.AddSeqMemPortAnnotation",
      name = "user_input",
      input = true,
      width = 1
    }
  ]
} {
  firrtl.layer @A bind {}

  // expected-error @below {{cannot have ports added to it}}
  firrtl.memmodule @mem(
    in W0_addr: !firrtl.uint<1>,
    in W0_en: !firrtl.uint<1>,
    in W0_clk: !firrtl.clock,
    in W0_data: !firrtl.uint<1>
  ) attributes {
    dataWidth = 1 : ui32,
    depth = 2 : ui64,
    extraPorts = [],
    maskBits = 1 : ui32,
    numReadPorts = 0 : ui32,
    numReadWritePorts = 0 : ui32,
    numWritePorts = 1 : ui32,
    readLatency = 1 : ui32,
    writeLatency = 1 : ui32
  }

  firrtl.module @Foo() {
    %0:4 = firrtl.instance mem @mem(
      in W0_addr: !firrtl.uint<1>,
      in W0_en: !firrtl.uint<1>,
      in W0_clk: !firrtl.clock,
      in W0_data: !firrtl.uint<1>
    )
  }

  firrtl.module @LayerBlock() {
    // expected-note @below {{the innermost layer block is here}}
    firrtl.layerblock @A {
      // expected-note @below {{this instance is under a layer block}}
      %0:4 = firrtl.instance mem @mem(
        in W0_addr: !firrtl.uint<1>,
        in W0_en: !firrtl.uint<1>,
        in W0_clk: !firrtl.clock,
        in W0_data: !firrtl.uint<1>
      )
    }
    firrtl.instance foo @Foo()
  }

}

// -----

// This has a memory that is instantiated under the design-under-test _and_
// under a module that is under a layer block.
firrtl.circuit "LayerBlock" attributes {
  annotations = [
    {
      class = "sifive.enterprise.firrtl.AddSeqMemPortAnnotation",
      name = "user_input",
      input = true,
      width = 1
    }
  ]
} {
  firrtl.layer @A bind {}

  // expected-error @below {{cannot have ports added to it}}
  firrtl.memmodule @mem(
    in W0_addr: !firrtl.uint<1>,
    in W0_en: !firrtl.uint<1>,
    in W0_clk: !firrtl.clock,
    in W0_data: !firrtl.uint<1>
  ) attributes {
    dataWidth = 1 : ui32,
    depth = 2 : ui64,
    extraPorts = [],
    maskBits = 1 : ui32,
    numReadPorts = 0 : ui32,
    numReadWritePorts = 0 : ui32,
    numWritePorts = 1 : ui32,
    readLatency = 1 : ui32,
    writeLatency = 1 : ui32
  }

  firrtl.module @Foo() {
    %0:4 = firrtl.instance mem @mem(
      in W0_addr: !firrtl.uint<1>,
      in W0_en: !firrtl.uint<1>,
      in W0_clk: !firrtl.clock,
      in W0_data: !firrtl.uint<1>
    )
  }

  firrtl.module @Bar() {
    // expected-note @below {{this instance is inside a module that is instantiated outside the design}}
    %0:4 = firrtl.instance mem @mem(
      in W0_addr: !firrtl.uint<1>,
      in W0_en: !firrtl.uint<1>,
      in W0_clk: !firrtl.clock,
      in W0_data: !firrtl.uint<1>
    )
  }

  firrtl.module @LayerBlock() {
    firrtl.layerblock @A {
      firrtl.instance bar @Bar()
    }
    firrtl.instance foo @Foo()
  }

}

// -----

// Test that a memory that is instantiated under the design-under-test (DUT) and
// _not_ under the DUT will error.
//
// See: https://github.com/llvm/circt/issues/7620
firrtl.circuit "Foo" attributes {
  annotations = [
    {
      class = "sifive.enterprise.firrtl.AddSeqMemPortAnnotation",
      name = "user_input",
      input = true,
      width = 1
    }
  ]
} {
  firrtl.layer @A bind {}

  // expected-error @below {{cannot have ports added to it}}
  firrtl.memmodule @mem(
    in waddr: !firrtl.uint<1>,
    in wen: !firrtl.uint<1>,
    in wclk: !firrtl.clock,
    in wdata: !firrtl.uint<1>
  ) attributes {
    dataWidth = 1 : ui32,
    depth = 2 : ui64,
    extraPorts = [],
    maskBits = 1 : ui32,
    numReadPorts = 0 : ui32,
    numReadWritePorts = 0 : ui32,
    numWritePorts = 1 : ui32,
    readLatency = 1 : ui32,
    writeLatency = 1 : ui32
  }

  firrtl.module @Bar() attributes {
    annotations = [
      {class = "sifive.enterprise.firrtl.MarkDUTAnnotation"}
    ]
  } {
    %0:4 = firrtl.instance mem @mem(
      in waddr: !firrtl.uint<1>,
      in wen: !firrtl.uint<1>,
      in wclk: !firrtl.clock,
      in wdata: !firrtl.uint<1>
    )
  }

  firrtl.module @Foo() {
    firrtl.instance bar @Bar()
    // expected-note @below {{this instance is inside a module that is instantiated outside the design}}
    %0:4 = firrtl.instance mem @mem(
      in waddr: !firrtl.uint<1>,
      in wen: !firrtl.uint<1>,
      in wclk: !firrtl.clock,
      in wdata: !firrtl.uint<1>
    )
  }

}
