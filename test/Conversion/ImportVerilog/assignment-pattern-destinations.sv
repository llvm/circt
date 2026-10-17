// RUN: circt-verilog --import-only %s | FileCheck %s
// RUN: circt-verilog %s -o /dev/null
// REQUIRES: slang
// UNSUPPORTED: valgrind

// Continuous assignments use the same declaration order as output connections.
// CHECK-LABEL: moore.module @continuous(
// CHECK: %[[VALUES:.*]] = moore.variable : <l2>
// CHECK: %[[LOW:.*]] = moore.extract_ref %[[VALUES]] from 0
// CHECK: %[[HIGH:.*]] = moore.extract_ref %[[VALUES]] from 1
// CHECK: %[[BITS:.*]] = moore.read %bits{{(_[0-9]+)?}} : <uarray<2 x l1>>
// CHECK-NEXT: %[[FIRST:.*]] = moore.extract %[[BITS]] from 1
// CHECK-NEXT: moore.assign %[[LOW]], %[[FIRST]] : l1
// CHECK-NEXT: %[[SECOND:.*]] = moore.extract %[[BITS]] from 0
// CHECK-NEXT: moore.assign %[[HIGH]], %[[SECOND]] : l1
module continuous(output logic [1:0] values);
  wire bits[2];
  assign bits = '{1'b0, 1'b1};
  assign '{values[0], values[1]} = bits;
endmodule

// Procedural assignments emit blocking writes, with one read of the RHS.
// CHECK-LABEL: moore.module @combinational(
// CHECK: moore.procedure always_comb {
// CHECK: %[[LOW:.*]] = moore.extract_ref %values from 0
// CHECK: %[[HIGH:.*]] = moore.extract_ref %values from 1
// CHECK: %[[BITS:.*]] = moore.read %bits{{(_[0-9]+)?}} : <uarray<2 x l1>>
// CHECK-NEXT: %[[FIRST:.*]] = moore.extract %[[BITS]] from 1
// CHECK-NEXT: moore.blocking_assign %[[LOW]], %[[FIRST]] : l1
// CHECK-NEXT: %[[SECOND:.*]] = moore.extract %[[BITS]] from 0
// CHECK-NEXT: moore.blocking_assign %[[HIGH]], %[[SECOND]] : l1
module combinational(input logic bits[2], output logic [1:0] values);
  always_comb '{values[0], values[1]} = bits;
endmodule

// Nested patterns preserve bounds, signed extension, and truncation.
// CHECK-LABEL: moore.module @nested_combinational(
// CHECK: moore.procedure always_comb {
// CHECK: %[[PART:.*]] = moore.extract_ref %y from 0
// CHECK: %[[A:.*]] = moore.read %a{{(_[0-9]+)?}} : <uarray<2 x uarray<2 x l4>>>
// CHECK-NEXT: %[[ROW0:.*]] = moore.extract %[[A]] from 1
// CHECK-NEXT: %[[V0:.*]] = moore.extract %[[ROW0]] from 1
// CHECK-NEXT: %[[EXT:.*]] = moore.sext %[[V0]] : l4 -> l8
// CHECK-NEXT: moore.blocking_assign %x, %[[EXT]] : l8
// CHECK-NEXT: %[[V1:.*]] = moore.extract %[[ROW0]] from 0
// CHECK-NEXT: moore.blocking_assign %[[PART]], %[[V1]] : l4
// CHECK-NEXT: %[[ROW1:.*]] = moore.extract %[[A]] from 0
// CHECK-NEXT: %[[V2:.*]] = moore.extract %[[ROW1]] from 1
// CHECK-NEXT: %[[TRUNC:.*]] = moore.trunc %[[V2]] : l4 -> l2
// CHECK-NEXT: moore.blocking_assign %z, %[[TRUNC]] : l2
// CHECK-NEXT: %[[V3:.*]] = moore.extract %[[ROW1]] from 0
// CHECK-NEXT: moore.blocking_assign %w, %[[V3]] : l4
module nested_combinational(input logic signed [3:0] a[-2:-1][7:6],
                            output logic signed [7:0] x,
                            output logic [7:0] y,
                            output logic [1:0] z,
                            output logic [3:0] w);
  always_comb begin
    y = '0;
    '{'{x, y[3:0]}, '{z, w}} = a;
  end
endmodule

// Snapshot the aggregate before either write so this really swaps the elements.
// CHECK-LABEL: moore.module @swap_elements(
// CHECK: moore.procedure always {
// CHECK: %[[LEFT:.*]] = moore.extract_ref %a from 0
// CHECK: %[[RIGHT:.*]] = moore.extract_ref %a from 1
// CHECK: %[[A:.*]] = moore.read %a : <uarray<2 x l8>>
// CHECK-NEXT: %[[FIRST:.*]] = moore.extract %[[A]] from 1
// CHECK-NEXT: moore.blocking_assign %[[LEFT]], %[[FIRST]] : l8
// CHECK-NEXT: %[[SECOND:.*]] = moore.extract %[[A]] from 0
// CHECK-NEXT: moore.blocking_assign %[[RIGHT]], %[[SECOND]] : l8
module swap_elements(input logic clk, output logic [7:0] a[2]);
  always @(posedge clk) '{a[1], a[0]} = a;
endmodule

// All destination indices must be resolved before the first blocking write.
// CHECK-LABEL: moore.module @indexed_destinations(
// CHECK: moore.procedure always_comb {
// CHECK: %[[INDEX:.*]] = moore.read %index : <l1>
// CHECK: %[[DEST:.*]] = moore.dyn_extract_ref %values
// CHECK: %[[BITS:.*]] = moore.read %bits{{(_[0-9]+)?}} : <uarray<2 x l1>>
// CHECK: moore.blocking_assign %index,
// CHECK: moore.blocking_assign %[[DEST]],
module indexed_destinations(input logic bits[2],
                            output logic index, output logic values[2]);
  always_comb '{index, values[index]} = bits;
endmodule

// Struct fields use field order and retain their individual signedness.
typedef struct { logic [2:0] a; logic signed [3:0] b; } pair_t;
// CHECK-LABEL: moore.module @continuous_struct(
// CHECK: %[[P:.*]] = moore.read %p_0 : <ustruct<{a: l3, b: l4}>>
// CHECK-NEXT: %[[A:.*]] = moore.struct_extract %[[P]], "a"
// CHECK-NEXT: moore.assign %a, %[[A]] : l3
// CHECK-NEXT: %[[B:.*]] = moore.struct_extract %[[P]], "b"
// CHECK-NEXT: %[[EXT:.*]] = moore.sext %[[B]] : l4 -> l8
// CHECK-NEXT: moore.assign %b, %[[EXT]] : l8
module continuous_struct(input pair_t p, output logic [2:0] a,
                         output logic signed [7:0] b);
  assign '{a, b} = p;
endmodule

// The same delay applies to every destination.
// CHECK-LABEL: moore.module @continuous_delay(
// CHECK: %[[LOW:.*]] = moore.extract_ref %values from 0
// CHECK: %[[HIGH:.*]] = moore.extract_ref %values from 1
// CHECK: %[[BITS:.*]] = moore.read %bits{{(_[0-9]+)?}} : <uarray<2 x l1>>
// CHECK: %[[DELAY:.*]] = moore.constant_time
// CHECK: %[[FIRST:.*]] = moore.extract %[[BITS]] from 1
// CHECK-NEXT: moore.delayed_assign %[[LOW]], %[[FIRST]], %[[DELAY]]
// CHECK-NEXT: %[[SECOND:.*]] = moore.extract %[[BITS]] from 0
// CHECK-NEXT: moore.delayed_assign %[[HIGH]], %[[SECOND]], %[[DELAY]]
module continuous_delay(input logic bits[2], output wire [1:0] values);
  assign #1 '{values[0], values[1]} = bits;
endmodule

// Procedural timing and assignment kind are preserved for every destination.
// CHECK-LABEL: moore.module @nonblocking_delay(
// CHECK: %[[LOW:.*]] = moore.extract_ref %values from 0
// CHECK: %[[HIGH:.*]] = moore.extract_ref %values from 1
// CHECK: %[[BITS:.*]] = moore.read %bits{{(_[0-9]+)?}} : <uarray<2 x l1>>
// CHECK: %[[DELAY:.*]] = moore.constant_time
// CHECK: %[[FIRST:.*]] = moore.extract %[[BITS]] from 1
// CHECK-NEXT: moore.delayed_nonblocking_assign %[[LOW]], %[[FIRST]], %[[DELAY]]
// CHECK-NEXT: %[[SECOND:.*]] = moore.extract %[[BITS]] from 0
// CHECK-NEXT: moore.delayed_nonblocking_assign %[[HIGH]], %[[SECOND]], %[[DELAY]]
module nonblocking_delay(input logic clk, input logic bits[2],
                         output logic [1:0] values);
  always @(posedge clk) '{values[0], values[1]} <= #1 bits;
endmodule
