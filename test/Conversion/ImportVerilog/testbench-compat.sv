// RUN: circt-verilog --ir-moore %s | FileCheck %s
// REQUIRES: slang
//
// Internal issue in Slang v3 about jump depending on uninitialised value.
// UNSUPPORTED: valgrind

// CHECK-LABEL: moore.module @TaskReturn()
module TaskReturn;
  // CHECK: moore.procedure initial {
  // CHECK-NEXT:   moore.call_coroutine @t() : () -> ()
  // CHECK-NEXT:   moore.return
  // CHECK-NEXT: }
  initial t();

  // CHECK-LABEL: moore.coroutine private @t() {
  // CHECK-NEXT:   moore.return
  // CHECK-NEXT: }
  task automatic t();
    return;
  endtask
endmodule
