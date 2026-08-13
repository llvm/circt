// RUN: circt-opt --pass-pipeline='builtin.module(lower-firrtl-to-hw{lower-to-core=true})' --verify-diagnostics --split-input-file %s

firrtl.circuit "fflush_unsupported" {
  firrtl.module @fflush_unsupported(
      in %clock: !firrtl.clock,
      in %enable: !firrtl.uint<1>) {
    // expected-error @+2 {{'firrtl.fflush' op lower-to-core does not support firrtl.fflush yet}}
    // expected-error @below {{'firrtl.fflush' op LowerToHW couldn't handle this operation}}
    firrtl.fflush %clock, %enable, "out.txt"()
        : !firrtl.clock, !firrtl.uint<1>
    firrtl.skip
  }
}

// -----

firrtl.circuit "force_unsupported" {
  firrtl.module @force_unsupported(in %in: !firrtl.uint<42>) {
    %foo = firrtl.verbatim.wire "foo" : () -> !firrtl.uint<42>
    // expected-error @+2 {{'firrtl.force' op lower-to-core does not support firrtl.force}}
    // expected-error @below {{'firrtl.force' op LowerToHW couldn't handle this operation}}
    firrtl.force %foo, %in : !firrtl.uint<42>, !firrtl.uint<42>
  }
}

// -----

firrtl.circuit "ref_force_unsupported" {
  firrtl.module @ref_force_unsupported(
      in %clock: !firrtl.clock,
      in %enable: !firrtl.uint<1>,
      in %x: !firrtl.uint<4>) {
    %w, %w_ref = firrtl.wire forceable : !firrtl.uint<4>, !firrtl.rwprobe<uint<4>>
    // expected-error @+2 {{'firrtl.ref.force' op lower-to-core does not support firrtl.ref.force}}
    // expected-error @below {{'firrtl.ref.force' op LowerToHW couldn't handle this operation}}
    firrtl.ref.force %clock, %enable, %w_ref, %x
        : !firrtl.clock, !firrtl.uint<1>, !firrtl.rwprobe<uint<4>>, !firrtl.uint<4>
    firrtl.skip
  }
}

// -----

firrtl.circuit "ref_force_initial_unsupported" {
  firrtl.module @ref_force_initial_unsupported(
      in %enable: !firrtl.uint<1>,
      in %x: !firrtl.uint<4>) {
    %w, %w_ref = firrtl.wire forceable : !firrtl.uint<4>, !firrtl.rwprobe<uint<4>>
    // expected-error @+2 {{'firrtl.ref.force_initial' op lower-to-core does not support firrtl.ref.force_initial}}
    // expected-error @below {{'firrtl.ref.force_initial' op LowerToHW couldn't handle this operation}}
    firrtl.ref.force_initial %enable, %w_ref, %x
        : !firrtl.uint<1>, !firrtl.rwprobe<uint<4>>, !firrtl.uint<4>
    firrtl.skip
  }
}

// -----

firrtl.circuit "ref_release_unsupported" {
  firrtl.module @ref_release_unsupported(
      in %clock: !firrtl.clock,
      in %enable: !firrtl.uint<1>) {
    %w, %w_ref = firrtl.wire forceable : !firrtl.uint<4>, !firrtl.rwprobe<uint<4>>
    // expected-error @+2 {{'firrtl.ref.release' op lower-to-core does not support firrtl.ref.release}}
    // expected-error @below {{'firrtl.ref.release' op LowerToHW couldn't handle this operation}}
    firrtl.ref.release %clock, %enable, %w_ref
        : !firrtl.clock, !firrtl.uint<1>, !firrtl.rwprobe<uint<4>>
    firrtl.skip
  }
}

// -----

firrtl.circuit "ref_release_initial_unsupported" {
  firrtl.module @ref_release_initial_unsupported(in %enable: !firrtl.uint<1>) {
    %w, %w_ref = firrtl.wire forceable : !firrtl.uint<4>, !firrtl.rwprobe<uint<4>>
    // expected-error @+2 {{'firrtl.ref.release_initial' op lower-to-core does not support firrtl.ref.release_initial}}
    // expected-error @below {{'firrtl.ref.release_initial' op LowerToHW couldn't handle this operation}}
    firrtl.ref.release_initial %enable, %w_ref
        : !firrtl.uint<1>, !firrtl.rwprobe<uint<4>>
    firrtl.skip
  }
}

// -----

firrtl.circuit "attach_unsupported" {
  firrtl.extmodule @AnalogIn(in a: !firrtl.analog<8>)
  firrtl.extmodule @AnalogOut(out a: !firrtl.analog<8>)

  firrtl.module @attach_unsupported() {
    %in = firrtl.instance in @AnalogIn(in a: !firrtl.analog<8>)
    %out = firrtl.instance out @AnalogOut(out a: !firrtl.analog<8>)
    // expected-error @+2 {{'firrtl.attach' op lower-to-core does not support firrtl.attach that requires SV lowering}}
    // expected-error @below {{'firrtl.attach' op LowerToHW couldn't handle this operation}}
    firrtl.attach %in, %out : !firrtl.analog<8>, !firrtl.analog<8>
  }
}

// -----

firrtl.circuit "Top" {
  sv.macro.decl @targets$Opt$FPGA
  sv.macro.decl @targets$Opt$Top$inst
  firrtl.option @Opt {
    firrtl.option_case @FPGA { case_macro = @targets$Opt$FPGA }
  }

  firrtl.module private @Default(in %in: !firrtl.uint<8>, out %p: !firrtl.probe<uint<8>>) {
    %ref = firrtl.ref.send %in : !firrtl.uint<8>
    firrtl.ref.define %p, %ref : !firrtl.probe<uint<8>>
  }

  firrtl.module private @FPGA(in %in: !firrtl.uint<8>, out %p: !firrtl.probe<uint<8>>) {
    %ref = firrtl.ref.send %in : !firrtl.uint<8>
    firrtl.ref.define %p, %ref : !firrtl.probe<uint<8>>
  }

  firrtl.module @Top(in %in: !firrtl.uint<8>) {
    // expected-error @+2 {{'firrtl.instance_choice' op lower-to-core does not support probe output port p on instance_choice}}
    // expected-error @below {{'firrtl.instance_choice' op LowerToHW couldn't handle this operation}}
    %inst_in, %p = firrtl.instance_choice inst {instance_macro = @targets$Opt$Top$inst} @Default alternatives @Opt
                   { @FPGA -> @FPGA } (in in: !firrtl.uint<8>, out p: !firrtl.probe<uint<8>>)
    firrtl.connect %inst_in, %in : !firrtl.uint<8>, !firrtl.uint<8>
  }
}
