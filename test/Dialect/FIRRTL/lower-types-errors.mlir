// RUN: circt-opt -pass-pipeline='builtin.module(firrtl.circuit(firrtl-lower-types{preserve-aggregate=vec}))' %s --verify-diagnostics --split-input-file

// Check diagnostic when attempting to lower something with symbols on it.
firrtl.circuit "InnerSym" {
  firrtl.module @InnerSym(
  // expected-error @below {{unable to lower due to symbol "x" with target not preserved by lowering}}
    in %x: !firrtl.bundle<a: uint<5>, b: uint<3>>
      sym @x
    ) { }
}

// -----

firrtl.circuit "InstanceChoiceError" {
  firrtl.option @Platform {
    firrtl.option_case @FPGA
  }

  firrtl.module private @Target(
    in %in: !firrtl.vector<uint<8>, 2>,
    out %out: !firrtl.vector<uint<8>, 2>
  ) attributes {convention = #firrtl<convention internal>} { }

  firrtl.module public @PublicTarget(
    in %in: !firrtl.vector<uint<8>, 2>,
    out %out: !firrtl.vector<uint<8>, 2>
  ) attributes {convention = #firrtl<convention scalarized>}{ }

  firrtl.module @InstanceChoiceError() {
    // expected-error @below {{instance_choice has different preservation modes for different modules}}
    %inst_in, %inst_out = firrtl.instance_choice inst sym @sym @Target alternatives @Platform {
      @FPGA -> @PublicTarget
    } (in in: !firrtl.vector<uint<8>, 2>, out out: !firrtl.vector<uint<8>, 2>)
  }
}
