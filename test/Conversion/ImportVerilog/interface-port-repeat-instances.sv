// RUN: circt-verilog --ir-moore %s | FileCheck %s

// Interface-port connections on NON-CANONICAL instance bodies: a module with
// an interface port instantiated more than once (same specialization) carries
// distinct InterfacePortSymbols per instance body for the same source
// declaration. Connections must resolve to the canonical flattened ports via
// the port's syntax node, for plain instance connections as well as for
// element selects of a local interface instance array.

interface bus_if;
  logic req;
  logic ack;
endinterface

module leaf(bus_if bus);
  assign bus.ack = bus.req;
endmodule

// CHECK-LABEL: moore.module @top
module top;
  bus_if b0();
  bus_if b1();
  bus_if barr[2]();
  // CHECK: moore.instance "u0" @leaf
  leaf u0(.bus(b0));
  // CHECK: moore.instance "u1" @leaf
  leaf u1(.bus(b1));
  // CHECK: moore.instance "u2" @leaf
  leaf u2(.bus(barr[1]));
endmodule
