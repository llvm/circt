// RUN: circt-opt --lower-sim-to-sv %s | FileCheck %s

// CHECK: sv.func private @"__circt_lib_logging::FileDescriptor::get"
// CHECK: sv.macro.decl @__CIRCT_LIB_LOGGING
// CHECK: emit.fragment @CIRCT_LIB_LOGGING_FRAGMENT

hw.module @flush_file(in %clk : !seq.clock) {
  // CHECK-LABEL: hw.module @flush_file
  %lit = sim.fmt.literal "file"
  sim.triggered %clk {
    // CHECK: %[[FD:.+]] = sv.func.call.procedural @"__circt_lib_logging::FileDescriptor::get"(%{{.+}}) : (!hw.string) -> i32
    %file = sim.get_file %lit
    // CHECK: sv.fflush fd %[[FD]]
    sim.flush %file
  }
}

hw.module @flush_stdout_stderr(in %clk : !seq.clock) {
  // CHECK-LABEL: hw.module @flush_stdout_stderr
  // CHECK-DAG: %[[STDOUT:.+]] = hw.constant -2147483647 : i32
  // CHECK-DAG: %[[STDERR:.+]] = hw.constant -2147483646 : i32
  %stdout = sim.stdout_stream
  %stderr = sim.stderr_stream
  sim.triggered %clk {
    // CHECK: sv.fflush fd %[[STDOUT]]
    sim.flush %stdout
    // CHECK: sv.fflush fd %[[STDERR]]
    sim.flush %stderr
  }
}
