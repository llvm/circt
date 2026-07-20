// RUN: circt-verilog --ir-moore %s | FileCheck %s

// Modport-qualified interface-ARRAY ports (`bus_if.slave bus[2]`) are
// flattened once per element (prefix `<port>_<i>_`), so element connections
// to child instances resolve through the per-element interface lowering
// (previously: `interface instance was not expanded`).

interface bus_if;
  logic req;
  logic ack;
  modport slave(input req, output ack);
endinterface

module leaf(bus_if.slave bus);
  assign bus.ack = bus.req;
endmodule

// CHECK: moore.module private @mid
// CHECK-SAME: bus_0_req
// CHECK-SAME: bus_1_req
module mid(bus_if.slave bus[2]);
  // CHECK: moore.instance "u0" @leaf
  leaf u0(.bus(bus[0]));
  // CHECK: moore.instance "u1" @leaf
  leaf u1(.bus(bus[1]));
endmodule

// CHECK-LABEL: moore.module @top
module top;
  bus_if barr[2]();
  mid u_mid(.bus(barr));
endmodule
