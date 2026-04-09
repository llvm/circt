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
