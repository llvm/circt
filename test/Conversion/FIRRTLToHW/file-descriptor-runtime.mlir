// RUN: circt-opt -pass-pipeline="builtin.module(lower-firrtl-to-hw)" %s | FileCheck %s

// The file descriptor runtime must use the resolved macro symbol name when the
// requested Verilog macro name collides with an existing symbol.
firrtl.circuit "FileDescriptorRuntime" {
  firrtl.module @SYNTHESIS() {
    firrtl.skip
  }

  // CHECK:      sv.func private @"__circt_lib_logging::FileDescriptor::get"
  // CHECK:      sv.macro.decl @__CIRCT_LIB_LOGGING
  // CHECK-NEXT: sv.macro.decl @SYNTHESIS_0["SYNTHESIS"]
  // CHECK-NEXT: emit.fragment @CIRCT_LIB_LOGGING_FRAGMENT {
  // CHECK-NEXT:   sv.ifdef @SYNTHESIS_0 {
  // CHECK-NEXT:   } else {
  // CHECK-NEXT:     sv.ifdef @__CIRCT_LIB_LOGGING {
  // CHECK:      hw.module @SYNTHESIS()
  firrtl.module @FileDescriptorRuntime(
      in %clock: !firrtl.clock, in %enable: !firrtl.uint<1>,
      in %file: !firrtl.uint<4>) {
    %time = firrtl.fstring.time : !firrtl.fstring
    firrtl.fprintf %clock, %enable, "out%d.txt"(%file), "[{{}}]\0A"(%time)
        : !firrtl.clock, !firrtl.uint<1>, !firrtl.uint<4>, !firrtl.fstring
    firrtl.skip
  }
}
