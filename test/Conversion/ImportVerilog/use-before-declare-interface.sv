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

// CHECK-LABEL: moore.module @ForwardInterfaceLoopGenerate(
module ForwardInterfaceLoopGenerate(output logic [1:0] y);
  for (genvar i = 0; i < 2; ++i) begin : lane
    // CHECK-DAG: %lane_0.bus_sig = moore.variable : <l1>
    // CHECK-DAG: %lane_1.bus_sig = moore.variable : <l1>
    // CHECK: moore.blocking_assign %lane_0.bus_sig
    // CHECK: moore.assign {{.*}} : l1
    // CHECK: moore.blocking_assign %lane_1.bus_sig
    // CHECK: moore.assign {{.*}} : l1
    initial bus.sig = bit'(i);

    assign y[i] = bus.sig;
    simple_if bus(clk);
    logic clk;
  end
endmodule

// CHECK-LABEL: moore.module @ForwardInterfaceInitializer(
module ForwardInterfaceInitializer(output logic y);
  // CHECK-DAG: %captured = moore.variable {{.*}} : <l1>
  // CHECK-DAG: %bus_sig = moore.variable : <l1>
  // CHECK-DAG: moore.read %bus_sig : <l1>
  logic captured = bus.sig;

  // CHECK-DAG: moore.read %captured : <l1>
  // CHECK: moore.output {{.*}} : !moore.l1
  assign y = captured;

  simple_if bus(clk);
  logic clk;
endmodule
