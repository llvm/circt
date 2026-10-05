// RUN: circt-opt --firrtl-lower-classes --split-input-file --verify-diagnostics %s

firrtl.circuit "NoBodyMem" {
  // expected-error @below {{cannot lower module to an OM class: module has no body}}
  firrtl.memmodule @MemNoBody(
    in W0_clk: !firrtl.clock
  ) attributes {
    dataWidth = 8 : ui32,
    depth = 8 : ui64,
    extraPorts = [],
    maskBits = 1 : ui32,
    numReadPorts = 0 : ui32,
    numReadWritePorts = 0 : ui32,
    numWritePorts = 1 : ui32,
    readLatency = 1 : ui32,
    writeLatency = 1 : ui32
  }

  firrtl.module @NoBodyMem(in %clk: !firrtl.clock, out %prop: !firrtl.string) {
    %mem_clk = firrtl.instance mem @MemNoBody(in W0_clk: !firrtl.clock)
    firrtl.matchingconnect %mem_clk, %clk : !firrtl.clock
    %s = firrtl.string "hello"
    firrtl.propassign %prop, %s : !firrtl.string
  }
}

firrtl.circuit "NoBodyInt" {
  // expected-error @below {{cannot lower module to an OM class: module has no body}}
  firrtl.intmodule @IntNoBody(in clk: !firrtl.clock) attributes {intrinsic = "circt_test_intrinsic"}

  firrtl.module @NoBodyInt(in %clk: !firrtl.clock, out %prop: !firrtl.string) {
    %i_clk = firrtl.instance i @IntNoBody(in clk: !firrtl.clock)
    firrtl.matchingconnect %i_clk, %clk : !firrtl.clock
    %s = firrtl.string "hello"
    firrtl.propassign %prop, %s : !firrtl.string
  }
}

// CHECK-LABEL: firrtl.circuit "WithBody"
firrtl.circuit "WithBody" {
  // The property ports are erased from the module.
  // CHECK: firrtl.module @WithBody(in %clk: !firrtl.clock)
  firrtl.module @WithBody(in %clk: !firrtl.clock,
                          in %in: !firrtl.string,
                          out %out: !firrtl.string) {
    firrtl.propassign %out, %in : !firrtl.string
  }

  // The OM class has the base path plus the input property as parameters.
  // CHECK: om.class @WithBody_Class(%basepath: !om.basepath, %in: !om.string)
  // CHECK: om.class.fields
}

// CHECK-LABEL: firrtl.circuit "Top"
firrtl.circuit "Top" {
  firrtl.module @Top() {}

  // CHECK: om.class @Greeter(%basepath: !om.basepath)
  firrtl.class @Greeter(out %msg: !firrtl.string) {
    %0 = firrtl.string "hello"
    firrtl.propassign %msg, %0 : !firrtl.string
  }
  // CHECK: om.constant "hello" : !om.string
  // CHECK: om.class.fields
}
