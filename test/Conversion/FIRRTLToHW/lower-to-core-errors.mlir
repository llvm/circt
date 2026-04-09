// RUN: not circt-opt --pass-pipeline='builtin.module(lower-firrtl-to-hw{lower-to-core=true})' %s -split-input-file 2>&1 | FileCheck %s

firrtl.circuit "time_printf" {
  firrtl.module @time_printf(
      in %clock: !firrtl.clock,
      in %enable: !firrtl.uint<1>) {
    %time = firrtl.fstring.time : !firrtl.fstring
    firrtl.printf %clock, %enable, "{{}}\0A"(%time)
        : !firrtl.clock, !firrtl.uint<1>, !firrtl.fstring
    firrtl.skip
  }
}

// CHECK: error: lower-to-core does not support {{.*}} in printf
// CHECK: error: 'firrtl.printf' op LowerToHW couldn't handle this operation

// -----

firrtl.circuit "fprintf_unsupported" {
  firrtl.module @fprintf_unsupported(
      in %clock: !firrtl.clock,
      in %enable: !firrtl.uint<1>) {
    firrtl.fprintf %clock, %enable, "out.txt"(), "msg\0A"()
        : !firrtl.clock, !firrtl.uint<1>
    firrtl.skip
  }
}

// CHECK: error: 'firrtl.fprintf' op lower-to-core does not support firrtl.fprintf yet
// CHECK: error: 'firrtl.fprintf' op LowerToHW couldn't handle this operation

// -----

firrtl.circuit "fflush_unsupported" {
  firrtl.module @fflush_unsupported(
      in %clock: !firrtl.clock,
      in %enable: !firrtl.uint<1>) {
    firrtl.fflush %clock, %enable, "out.txt"()
        : !firrtl.clock, !firrtl.uint<1>
    firrtl.skip
  }
}

// CHECK: error: 'firrtl.fflush' op lower-to-core does not support firrtl.fflush yet
// CHECK: error: 'firrtl.fflush' op LowerToHW couldn't handle this operation

// -----

firrtl.circuit "force_unsupported" {
  firrtl.module @force_unsupported(in %in: !firrtl.uint<42>) {
    %foo = firrtl.verbatim.wire "foo" : () -> !firrtl.uint<42>
    firrtl.force %foo, %in : !firrtl.uint<42>, !firrtl.uint<42>
  }
}

// CHECK: error: 'firrtl.force' op lower-to-core does not support firrtl.force
// CHECK: error: 'firrtl.force' op LowerToHW couldn't handle this operation

// -----

firrtl.circuit "ref_force_unsupported" {
  firrtl.module @ref_force_unsupported(
      in %clock: !firrtl.clock,
      in %enable: !firrtl.uint<1>,
      in %x: !firrtl.uint<4>) {
    %w, %w_ref = firrtl.wire forceable : !firrtl.uint<4>, !firrtl.rwprobe<uint<4>>
    firrtl.ref.force %clock, %enable, %w_ref, %x
        : !firrtl.clock, !firrtl.uint<1>, !firrtl.rwprobe<uint<4>>, !firrtl.uint<4>
    firrtl.skip
  }
}

// CHECK: error: 'firrtl.ref.force' op lower-to-core does not support firrtl.ref.force
// CHECK: error: 'firrtl.ref.force' op LowerToHW couldn't handle this operation

// -----

firrtl.circuit "ref_force_initial_unsupported" {
  firrtl.module @ref_force_initial_unsupported(
      in %enable: !firrtl.uint<1>,
      in %x: !firrtl.uint<4>) {
    %w, %w_ref = firrtl.wire forceable : !firrtl.uint<4>, !firrtl.rwprobe<uint<4>>
    firrtl.ref.force_initial %enable, %w_ref, %x
        : !firrtl.uint<1>, !firrtl.rwprobe<uint<4>>, !firrtl.uint<4>
    firrtl.skip
  }
}

// CHECK: error: 'firrtl.ref.force_initial' op lower-to-core does not support firrtl.ref.force_initial
// CHECK: error: 'firrtl.ref.force_initial' op LowerToHW couldn't handle this operation

// -----

firrtl.circuit "ref_release_unsupported" {
  firrtl.module @ref_release_unsupported(
      in %clock: !firrtl.clock,
      in %enable: !firrtl.uint<1>) {
    %w, %w_ref = firrtl.wire forceable : !firrtl.uint<4>, !firrtl.rwprobe<uint<4>>
    firrtl.ref.release %clock, %enable, %w_ref
        : !firrtl.clock, !firrtl.uint<1>, !firrtl.rwprobe<uint<4>>
    firrtl.skip
  }
}

// CHECK: error: 'firrtl.ref.release' op lower-to-core does not support firrtl.ref.release
// CHECK: error: 'firrtl.ref.release' op LowerToHW couldn't handle this operation

// -----

firrtl.circuit "ref_release_initial_unsupported" {
  firrtl.module @ref_release_initial_unsupported(in %enable: !firrtl.uint<1>) {
    %w, %w_ref = firrtl.wire forceable : !firrtl.uint<4>, !firrtl.rwprobe<uint<4>>
    firrtl.ref.release_initial %enable, %w_ref
        : !firrtl.uint<1>, !firrtl.rwprobe<uint<4>>
    firrtl.skip
  }
}

// CHECK: error: 'firrtl.ref.release_initial' op lower-to-core does not support firrtl.ref.release_initial
// CHECK: error: 'firrtl.ref.release_initial' op LowerToHW couldn't handle this operation

// -----

firrtl.circuit "attach_unsupported" {
  firrtl.extmodule @AnalogIn(in a: !firrtl.analog<8>)
  firrtl.extmodule @AnalogOut(out a: !firrtl.analog<8>)

  firrtl.module @attach_unsupported() {
    %in = firrtl.instance in @AnalogIn(in a: !firrtl.analog<8>)
    %out = firrtl.instance out @AnalogOut(out a: !firrtl.analog<8>)
    firrtl.attach %in, %out : !firrtl.analog<8>, !firrtl.analog<8>
  }
}

// CHECK: error: 'firrtl.attach' op lower-to-core does not support firrtl.attach that requires SV lowering
// CHECK: error: 'firrtl.attach' op LowerToHW couldn't handle this operation
