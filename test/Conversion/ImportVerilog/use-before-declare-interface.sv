// RUN: not circt-verilog --ir-moore %s 2>&1 | FileCheck %s --check-prefix=NOALLOW
// RUN: circt-verilog --ir-moore --allow-use-before-declare %s | FileCheck %s
// REQUIRES: slang
//
// Internal issue in Slang v3 about jump depending on uninitialised value.
// UNSUPPORTED: valgrind

// NOALLOW: identifier 'value' used before its declaration

interface simple_if(input logic clk);
  logic sig;
endinterface

interface passthrough_if(input logic clk);
  logic seen;
  assign seen = clk;
endinterface

// CHECK-LABEL: moore.module @ForwardInterfaceDirectMember(
module ForwardInterfaceDirectMember(output logic y);
  // The interface instance appears after both uses. In use-before-declare mode
  // ImportVerilog must still expand the instance early enough for `bus.sig` to
  // resolve in procedural and continuous assignments.
  // CHECK: %value = moore.variable : <i1>
  // CHECK: %bus_sig = moore.variable : <l1>
  // CHECK: moore.procedure initial {
  // CHECK:   moore.read %value : <i1>
  // CHECK:   moore.blocking_assign %bus_sig
  // CHECK: }
  initial bus.sig = value;

  // CHECK: moore.read %bus_sig : <l1>
  // CHECK: moore.output {{.*}} : !moore.l1
  assign y = bus.sig;

  simple_if bus(clk);
  logic clk;
  bit value;
endmodule

// CHECK-LABEL: moore.module @ForwardInterfaceBodyAssign(
module ForwardInterfaceBodyAssign(output logic y);
  // This covers the two-phase predeclaration order: the later `clk` variable
  // must be available before the earlier interface instance is expanded, since
  // the interface body's continuous assignment reads the connected port.
  // CHECK: %clk = moore.variable : <l1>
  // CHECK: %bus_seen = moore.assigned_variable
  // CHECK: moore.output %bus_seen : !moore.l1
  assign y = bus.seen;

  passthrough_if bus(clk);
  logic clk;
endmodule

// CHECK-LABEL: moore.module @ForwardInterfaceGenerate(
module ForwardInterfaceGenerate(output logic y);
  if (1) begin : g
    // CHECK: %g.value = moore.variable : <i1>
    // CHECK: %g.bus_sig = moore.variable : <l1>
    // CHECK: moore.procedure initial {
    // CHECK:   moore.read %g.value : <i1>
    // CHECK:   moore.blocking_assign %g.bus_sig
    // CHECK: }
    initial bus.sig = value;

    // CHECK: moore.read %g.bus_sig : <l1>
    assign y = bus.sig;

    simple_if bus(clk);
    logic clk;
    bit value;
  end
endmodule
