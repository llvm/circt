// RUN: circt-verilog --ir-moore %s | FileCheck %s
// REQUIRES: slang
// UNSUPPORTED: valgrind

// A `$stacktrace` in a procedural scope lowers to a display of a constant
// stack-trace message naming the surrounding procedural scope.
// CHECK-LABEL: moore.module @Top
module Top;
  initial begin
    // CHECK: moore.fmt.literal "Stack trace:\0A    <procedural scope>\0A    at
    // CHECK: moore.builtin.display
    $stacktrace;
  end
endmodule

// A subroutine that can reach `$stacktrace` gains a hidden caller-location
// string argument. Its body materializes the stack-trace message from the
// constant frame, the threaded caller string, and a trailing newline. The
// `initial` call site passes the caller location, so the produced IR is
// well-typed.
// CHECK-LABEL: func.func private @foo
// CHECK-SAME: !moore.string
// CHECK: moore.fmt.literal "Stack trace:\0A
// CHECK: moore.fmt.string
// CHECK: moore.builtin.display
module Caller;
  function automatic void foo();
    $stacktrace;
  endfunction
  initial foo();
endmodule
