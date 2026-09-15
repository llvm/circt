// RUN: circt-verilog --import-only --top=top %s -o /dev/null
// RUN: circt-verilog --ir-moore --top=top %s | FileCheck %s
// REQUIRES: slang

interface bus;
  logic valid;
  logic ready;
  modport sink(input valid, output ready);
endinterface

// Parameterized dimensions and both input and output modport members.
// CHECK-LABEL: moore.module private @dut(
// CHECK-SAME: in %b_0_valid : !moore.l1, out b_0_ready : !moore.l1,
// CHECK-SAME: in %b_1_valid : !moore.l1, out b_1_ready : !moore.l1,
// CHECK-SAME: out q : !moore.l2
// CHECK: %[[Q:.*]] = moore.concat %b_1_valid, %b_0_valid
// CHECK: moore.output %b_1_valid, %b_0_valid, %[[Q]]
module dut #(parameter PORTS = 2)(bus.sink b[PORTS], output logic [1:0] q);
  assign q = {b[1].valid, b[0].valid};
  assign b[0].ready = b[1].valid;
  assign b[1].ready = b[0].valid;
endmodule

// Forward an interface array through another module.
// CHECK-LABEL: moore.module private @wrapper(
// CHECK: %[[R0:.*]], %[[R1:.*]], %[[Q:.*]] = moore.instance "d" @dut(
// CHECK-SAME: b_0_valid: %b_0_valid: !moore.l1, b_1_valid: %b_1_valid: !moore.l1
// CHECK-SAME: -> (b_0_ready: !moore.l1, b_1_ready: !moore.l1, q: !moore.l2)
// CHECK: moore.output %[[R0]], %[[R1]], %[[Q]]
module wrapper(bus.sink b[2], output logic [1:0] q);
  dut #(.PORTS(2)) d(.b, .q);
endmodule

// Preserve scalar interface behavior, including selecting an array element.
// CHECK-LABEL: moore.module private @scalar(
// CHECK-SAME: in %b_valid : !moore.l1, out b_ready : !moore.l1, out q : !moore.l1
// CHECK: moore.output %b_valid, %b_valid
module scalar(bus.sink b, output logic q);
  assign q = b.valid;
  assign b.ready = b.valid;
endmodule

// No modport: each element's members must be passed as references.
// CHECK-LABEL: moore.module private @plain(
// CHECK-SAME: in %b_0_valid : !moore.ref<l1>, in %b_0_ready : !moore.ref<l1>,
// CHECK-SAME: in %b_1_valid : !moore.ref<l1>, in %b_1_ready : !moore.ref<l1>
// CHECK: %[[V1:.*]] = moore.read %b_1_valid
// CHECK: moore.assign %b_0_ready, %[[V1]]
// CHECK: %[[V0:.*]] = moore.read %b_0_valid
// CHECK: moore.assign %b_1_ready, %[[V0]]
module plain(bus b[2]);
  assign b[0].ready = b[1].valid;
  assign b[1].ready = b[0].valid;
endmodule

// Multiple dimensions with non-zero bounds and opposite directions at the
// connection. Slang rebases the connection to the formal port's dimensions.
// CHECK-LABEL: moore.module private @matrix(
// CHECK-SAME: in %b_3_6_valid : !moore.l1, out b_3_6_ready : !moore.l1,
// CHECK-SAME: in %b_3_7_valid : !moore.l1, out b_3_7_ready : !moore.l1,
// CHECK-SAME: in %b_4_6_valid : !moore.l1, out b_4_6_ready : !moore.l1,
// CHECK-SAME: in %b_4_7_valid : !moore.l1, out b_4_7_ready : !moore.l1,
// CHECK: %[[Q:.*]] = moore.concat %b_3_7_valid, %b_3_6_valid, %b_4_7_valid, %b_4_6_valid
// CHECK: moore.output %b_4_7_valid, %b_4_6_valid, %b_3_7_valid, %b_3_6_valid, %[[Q]]
module matrix(bus.sink b[3:4][7:6], output logic [3:0] q);
  assign q = {b[3][7].valid, b[3][6].valid,
              b[4][7].valid, b[4][6].valid};
  assign b[3][7].ready = b[4][6].valid;
  assign b[3][6].ready = b[4][7].valid;
  assign b[4][7].ready = b[3][6].valid;
  assign b[4][6].ready = b[3][7].valid;
endmodule

// CHECK-LABEL: moore.module @top(
// CHECK: %[[V0:.*]] = moore.assigned_variable %v0
// CHECK: moore.assigned_variable %w.b_0_ready
// CHECK: %[[V1:.*]] = moore.assigned_variable {{.*}} %v1
// CHECK: moore.assigned_variable {{.*}} %w.b_1_ready
// CHECK: %[[G81:.*]] = moore.assigned_variable {{.*}} %g81
// CHECK: moore.assigned_variable {{.*}} %matrix_d.b_4_7_ready
// CHECK: %[[G82:.*]] = moore.assigned_variable {{.*}} %g82
// CHECK: moore.assigned_variable {{.*}} %matrix_d.b_4_6_ready
// CHECK: %[[G91:.*]] = moore.assigned_variable {{.*}} %g91
// CHECK: moore.assigned_variable {{.*}} %matrix_d.b_3_7_ready
// CHECK: %[[G92:.*]] = moore.assigned_variable {{.*}} %g92
// CHECK: moore.assigned_variable {{.*}} %matrix_d.b_3_6_ready
// CHECK: moore.instance "w" @wrapper(b_0_valid: %[[V0]]: !moore.l1, b_1_valid: %[[V1]]: !moore.l1)
// CHECK: moore.instance "scalar_d" @scalar(
// CHECK: moore.instance "selected_d" @scalar(
// CHECK: moore.instance "plain_d" @plain(
// CHECK: moore.instance "matrix_d" @matrix(
// CHECK-SAME: b_3_6_valid: %[[G92]]: !moore.l1, b_3_7_valid: %[[G91]]: !moore.l1,
// CHECK-SAME: b_4_6_valid: %[[G82]]: !moore.l1, b_4_7_valid: %[[G81]]: !moore.l1
module top(input logic v0, v1, g91, g92, g81, g82,
           output logic [1:0] q, output logic s, output logic [3:0] m);
  bus b[2]();
  assign b[0].valid = v0;
  assign b[1].valid = v1;
  wrapper w(.b, .q);
  bus single();
  scalar scalar_d(.b(single), .q(s));
  bus selected[2]();
  scalar selected_d(.b(selected[1]), .q());
  bus p[2]();
  plain plain_d(.b(p));
  bus grid[9:8][1:2]();
  assign grid[9][1].valid = g91;
  assign grid[9][2].valid = g92;
  assign grid[8][1].valid = g81;
  assign grid[8][2].valid = g82;
  matrix matrix_d(.b(grid), .q(m));
endmodule
