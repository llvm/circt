// RUN: circt-translate --import-verilog %s | FileCheck %s
// RUN: circt-verilog --ir-moore %s
// REQUIRES: slang
// UNSUPPORTED: valgrind

// CHECK-LABEL: moore.module @ForkBlockDeclarations
module ForkBlockDeclarations;
  int i;
  int o;

  initial begin
    fork
      automatic int k = i;
      begin
        o = k;
      end
    join_none
  end
endmodule

// CHECK: moore.procedure initial
// CHECK:   [[I:%.+]] = moore.read {{%.+}} : <i32>
// CHECK:   [[K:%.+]] = moore.variable [[I]] : <i32>
// CHECK:   moore.fork join_none {
// CHECK:     [[K_VALUE:%.+]] = moore.read [[K]] : <i32>
// CHECK:     moore.blocking_assign {{%.+}}, [[K_VALUE]] : i32
// CHECK:     moore.complete
