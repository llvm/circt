// RUN: circt-verilog --ir-moore %s | FileCheck %s
// REQUIRES: slang
// UNSUPPORTED: valgrind

`timescale 1ms/1us

// CHECK-LABEL: moore.module @Top
module Top;
  initial begin
    // CHECK-NOT: unsupported system call `$printtimescale`
    // CHECK: moore.builtin.display
    $printtimescale;
  end
endmodule

`timescale 1us/1ns

module Child;
endmodule

// CHECK-LABEL: moore.module @Parent
module Parent;
  Child c();
  initial begin
    // CHECK-NOT: unsupported system call `$printtimescale`
    // CHECK: moore.builtin.display
    $printtimescale(c);
  end
endmodule
