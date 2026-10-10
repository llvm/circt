// RUN: circt-verilog --import-only --top=top %s -o /dev/null
// RUN: circt-verilog --ir-moore --top=top %s | FileCheck %s
// RUN: circt-verilog --import-only --top=gen_top %s -o /dev/null
// RUN: circt-verilog --ir-moore --top=gen_top %s | FileCheck %s --check-prefix=GEN
// REQUIRES: slang
// UNSUPPORTED: valgrind

// An uninitialized member does not invalidate an interface-array connection.
interface bus;
  logic base;
  modport sink(input base);
endinterface

// CHECK-LABEL: moore.module private @child(
// CHECK-SAME: in %intfs_0_base : !moore.l1, in %intfs_1_base : !moore.l1
// CHECK: moore.output %intfs_0_base : !moore.l1
module child(bus.sink intfs[2], output logic y);
  assign y = intfs[0].base;
endmodule

// CHECK-LABEL: moore.module @top(
// CHECK: %[[B0:.*]] = moore.variable : <l1>
// CHECK: %[[B1:.*]] = moore.variable {{.*}} : <l1>
// CHECK: %[[V0:.*]] = moore.read %[[B0]] : <l1>
// CHECK: %[[V1:.*]] = moore.read %[[B1]] : <l1>
// CHECK: moore.instance "c" @child(intfs_0_base: %[[V0]]: !moore.l1, intfs_1_base: %[[V1]]: !moore.l1)
module top(output logic y);
  bus b[2]();
  child c(b, y);
endmodule

interface gen_bus;
  wire base;
  modport bidir(inout base);
endinterface

// Generate indices must resolve to separate elements, with inout members
// passed as references so both reads and writes reach the connected net.
// GEN-LABEL: moore.module private @gen_child(
// GEN-SAME: in %intfs_0_base : !moore.ref<l1>, in %intfs_1_base : !moore.ref<l1>
// GEN: %[[X0:.*]] = moore.extract %x from 0 : l2 -> l1
// GEN: moore.assign %intfs_0_base, %[[X0]] : l1
// GEN: moore.read %intfs_0_base : <l1>
// GEN: %[[X1:.*]] = moore.extract %x from 1 : l2 -> l1
// GEN: moore.assign %intfs_1_base, %[[X1]] : l1
// GEN: moore.read %intfs_1_base : <l1>
module gen_child(gen_bus.bidir intfs[2], input logic [1:0] x,
                 output logic [1:0] y);
  for (genvar i = 0; i < 2; i++) begin : g
    assign intfs[i].base = x[i];
    assign y[i] = intfs[i].base;
  end
endmodule

// GEN-LABEL: moore.module @gen_top(
// GEN: %[[B0:.*]] = moore.net wire : <l1>
// GEN: %[[B1:.*]] = moore.net name "_base" wire : <l1>
// GEN: moore.instance "c" @gen_child(intfs_0_base: %[[B0]]: !moore.ref<l1>, intfs_1_base: %[[B1]]: !moore.ref<l1>, x: %x: !moore.l2)
module gen_top(input logic [1:0] x, output logic [1:0] y);
  gen_bus b[2]();
  gen_child c(b, x, y);
endmodule
