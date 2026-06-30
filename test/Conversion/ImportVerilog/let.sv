// RUN: circt-translate --import-verilog %s | FileCheck %s
// RUN: circt-verilog --ir-moore %s
// REQUIRES: slang
// UNSUPPORTED: valgrind

// CHECK-LABEL: moore.module @LetConstruct()
module LetConstruct;
  logic [3:0] a = 12;
  logic [3:0] b = 15;
  logic [3:0] c = 7;
  logic d;

  let op(x, y, z) = |((x | y) & z);

  initial begin
    // CHECK: moore.procedure initial
    // CHECK:   [[A:%.+]] = moore.read {{%.+}} : <l4>
    // CHECK:   [[B:%.+]] = moore.read {{%.+}} : <l4>
    // CHECK:   [[OR:%.+]] = moore.or [[A]], [[B]] : l4
    // CHECK:   [[C:%.+]] = moore.read {{%.+}} : <l4>
    // CHECK:   [[AND:%.+]] = moore.and [[OR]], [[C]] : l4
    // CHECK:   [[REDUCE:%.+]] = moore.reduce_or [[AND]] : l4 -> l1
    // CHECK:   moore.blocking_assign {{%.+}}, [[REDUCE]] : l1
    d = op(.x(a), .y(b), .z(c));
  end
endmodule
