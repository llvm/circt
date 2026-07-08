// RUN: circt-verilog --ir-moore %s | FileCheck %s

// Generic interface ports connected to an ARRAY of interface instances
// (`interface p[2]` connected to `if_t X[2]()`): each element is flattened
// with the port prefix `<port>_<i>_` and per-element accesses `p[K].member`
// resolve to the K-th element's flattened ports — NOT last-wins across
// elements (slang's canonical-body caching aliases the body member symbol
// across elements, so element ports resolve through the per-element
// interface lowering rather than the plain value-symbol table).

interface array_if;
  logic v;
  logic w;
endinterface

// The DUT writes a different value to each element and reads one back:
// distinct element ports must flow to distinct drives.
// CHECK-LABEL: moore.module private @dut
// CHECK-SAME: in %p_0_v : !moore.ref<l1>
// CHECK-SAME: in %p_0_w : !moore.ref<l1>
// CHECK-SAME: in %p_1_v : !moore.ref<l1>
// CHECK-SAME: in %p_1_w : !moore.ref<l1>
module dut (interface p[2], input logic a, input logic b, output logic y);
  // CHECK: moore.procedure always
  // CHECK: moore.nonblocking_assign %p_0_v, %{{.+}} : l1
  // CHECK: moore.nonblocking_assign %p_1_v, %{{.+}} : l1
  always @(a or b) begin
    p[0].v <= a;
    p[1].v <= b;
  end
  // Read-back rvalue path: y comes from element 1.
  // CHECK: moore.read %p_1_w
  assign y = p[1].w;
endmodule

// CHECK-LABEL: moore.module @ArrayIfTop
module ArrayIfTop (input logic a, input logic b, output logic y);
  array_if X[2] ();
  // The instantiation resolves each element port to the matching element
  // instance's expanded member storage.
  // CHECK: moore.instance "u_dut" @dut
  dut u_dut (.p(X), .a(a), .b(b), .y(y));
endmodule
