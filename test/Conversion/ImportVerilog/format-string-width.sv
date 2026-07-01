// RUN: circt-verilog --ir-moore %s | FileCheck %s
// REQUIRES: slang
// UNSUPPORTED: valgrind

// A field width on a `%s` specifier prints the string in a field of at least
// that many characters, right- or left-justified and space-padded
// (IEEE 1800-2017 § 21.2.1.2). Carried on `moore.fmt.string` as width /
// alignment / padding. (pulp-platform cheshire cva6 instr_trace_item.svh:368
// uses `%-36s`.)

// CHECK-LABEL: moore.module @FmtWidth(
module FmtWidth;
  string name;
  initial begin
    // CHECK: moore.fmt.string %{{.+}}, width 36, alignment left, padding space
    $display("%-36s", name);
    // CHECK: moore.fmt.string %{{.+}}, width 20, alignment right, padding space
    $display("%20s", name);
    // CHECK: moore.fmt.string %{{.+}}, width 42, alignment right, padding space
    $write("%42s", "foo");
  end
endmodule
