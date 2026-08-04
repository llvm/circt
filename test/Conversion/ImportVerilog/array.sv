// RUN: circt-verilog --ir-moore %s | FileCheck %s
// REQUIRES: slang

// Internal issue in Slang v3 about jump depending on uninitialised value.
// UNSUPPORTED: valgrind

//===---------------------------------------------------------------------===//
// Array element indexing
//===---------------------------------------------------------------------===//
// hw.array's indices are always ordered such that index 0 means the least
// significant element.
//
// SV types provide a mapping onto those elements, with [3:0] being the identity
// mapping and [0:3] being the reversed mapping.

// CHECK-LABEL: @array_get_dyn_packed_rev
// CHECK: %[[C:.*]] = moore.constant -1
// CHECK: %[[S:.*]] = moore.sub %[[C]], %idx
// CHECK: %[[X:.*]] = moore.dyn_extract %arg0 from %[[S]]
// CHECK-NEXT: moore.output %[[X]]
module array_get_dyn_packed_rev
    (input wire [0 : 3][15 : 0] arg0,
     input wire [1 : 0] idx,
     output wire [15 : 0] elem);
  assign elem = arg0[idx];
endmodule

// CHECK-LABEL: @array_get_dyn_unpacked
// CHECK: %[[C:.*]] = moore.constant -1
// CHECK: %[[S:.*]] = moore.sub %[[C]], %idx
// CHECK: %[[X:.*]] = moore.dyn_extract %arg0 from %[[S]]
// CHECK-NEXT: moore.output %[[X]]
module array_get_dyn_unpacked
    (input wire [15 : 0] arg0[4],
     input wire [1 : 0] idx,
     output wire [15 : 0] elem);
  assign elem = arg0[idx];
endmodule

// CHECK-LABEL: @array_get_dyn_unpacked_rev
// CHECK: %[[X:.*]] = moore.dyn_extract %arg0 from %idx
// CHECK-NEXT: moore.output %[[X]]
module array_get_dyn_unpacked_rev
    (input wire [15 : 0] arg0[3 : 0],
     input wire [1 : 0] idx,
     output wire [15 : 0] elem);
  assign elem = arg0[idx];
endmodule

// CHECK-LABEL: @array_get_static_packed
// CHECK: %[[X:.*]] = moore.extract %arg0 from 1
// CHECK-NEXT: moore.output %[[X]]
module array_get_static_packed
    (input wire [3 : 0][15 : 0] arg0,
     output wire [15 : 0] elem);
  assign elem = arg0[1];
endmodule

// CHECK-LABEL: @array_get_static_packed_rev
// CHECK: %[[X:.*]] = moore.extract %arg0 from 2
// CHECK-NEXT: moore.output %[[X]]
module array_get_static_packed_rev
    (input wire [0 : 3][15 : 0] arg0,
     output wire [15 : 0] elem);
  assign elem = arg0[1];
endmodule

// CHECK-LABEL: @array_get_static_unpacked
// CHECK: %[[X:.*]] = moore.extract %arg0 from 2
// CHECK-NEXT: moore.output %[[X]]
module array_get_static_unpacked
    (input wire [15 : 0] arg0[4],
     output wire [15 : 0] elem);
  assign elem = arg0[1];
endmodule

// CHECK-LABEL: @array_get_static_unpacked_rev
// CHECK: %[[X:.*]] = moore.extract %arg0 from 1
// CHECK-NEXT: moore.output %[[X]]
module array_get_static_unpacked_rev
    (input wire [15 : 0] arg0[3 : 0],
     output wire [15 : 0] elem);
  assign elem = arg0[1];
endmodule

// CHECK-LABEL: @array_get_static_unpacked_rev_rev
// CHECK: %[[X:.*]] = moore.extract %arg0 from 2
module array_get_static_unpacked_rev_rev
    (input wire [15 : 0] arg0[0 : 3],
     output wire [15 : 0] elem);
  assign elem = arg0[1];
endmodule

// CHECK-LABEL: @array_slice_dyn_packed
// CHECK: %[[X:.*]] = moore.dyn_extract %arg0 from %idx
// CHECK-NEXT: moore.output %[[X]]
module array_slice_dyn_packed
    (input wire [7 : 0][15 : 0] arg0,
     input wire [2 : 0] idx,
     output wire [3 : 0][15 : 0] slice);
  assign slice = arg0[idx +: 4];
endmodule

// CHECK-LABEL: @array_slice_dyn_packed_rev
// CHECK: %[[S:.*]] = moore.sub
// CHECK: %[[X:.*]] = moore.dyn_extract %arg0 from %[[S]]
// CHECK-NEXT: moore.output %[[X]]
module array_slice_dyn_packed_rev
    (input wire [0 : 7][15 : 0] arg0,
     input wire [2 : 0] idx,
     output wire [3 : 0][15 : 0] slice);
  assign slice = arg0[idx +: 4];
endmodule

// CHECK-LABEL: @array_slice_dyn_unpacked
// CHECK: %[[S:.*]] = moore.sub
// CHECK: %[[X:.*]] = moore.dyn_extract %arg0 from %[[S]]
// CHECK-NEXT: moore.output %[[X]]
module array_slice_dyn_unpacked
    (input wire [15 : 0] arg0[8],
     input wire [2 : 0] idx,
     output wire [15 : 0] slice[4]);
  assign slice = arg0[idx +: 4];
endmodule

// CHECK-LABEL: @array_slice_static_packed
// CHECK: %[[X:.*]] = moore.extract %arg0 from 1
// CHECK-NEXT: moore.output %[[X]]
module array_slice_static_packed
    (input wire [7 : 0][15 : 0] arg0,
     output wire [3 : 0][15 : 0] slice);
  assign slice = arg0[1 +: 4];
endmodule

// CHECK-LABEL: @array_slice_static_unpacked
// CHECK: %[[X:.*]] = moore.extract %arg0 from 3
// CHECK-NEXT: moore.output %[[X]]
module array_slice_static_unpacked
    (input wire [15 : 0] arg0[8],
     output wire [15 : 0] slice[4]);
  assign slice = arg0[1 +: 4];
endmodule

//===----------------------------------------------------------------------===//
// Array assignment
//===----------------------------------------------------------------------===//
//
// The assignment pattern '{...} populates an array starting from its leftmost
// bound, which is inferred from the type; `3` for `[3:0]`, `0` for `[0:3]`.
//
// The natural ordering for unpacked array assignment is little endian [0:N]:
//
//  7.4.2: "A declaration like `logic [7:0] b [3]` is equivalent to
//         `logic [7:0] b [0:2]`.

// CHECK-LABEL: @array_assign_packed
// CHECK: moore.array_create %d, %c, %b, %a
module array_assign_packed
    (input wire [15 : 0] a, b, c, d,
     output wire [3 : 0][15 : 0] out);
  assign out = '{d, c, b, a};
endmodule

// CHECK-LABEL: @array_assign_unpacked
// CHECK: moore.array_create %d, %c, %b, %a
module array_assign_unpacked
    (input wire [15 : 0] a, b, c, d,
     output wire [15 : 0] out[4]);
  assign out = '{d, c, b, a};
endmodule

// CHECK-LABEL: @array_assign_rev_packed
// CHECK: moore.array_create %a, %b, %c, %d
module array_assign_rev_packed
    (input wire [15 : 0] a, b, c, d,
     output wire [0 : 3][15 : 0] out);
  assign out = '{d, c, b, a};
endmodule

// CHECK-LABEL: @array_assign_rev_unpacked
// CHECK: moore.array_create %d, %c, %b, %a
module array_assign_rev_unpacked
    (input wire [15 : 0] a, b, c, d,
     output wire [15 : 0] out[3 : 0]);
  assign out = '{d, c, b, a};
endmodule

// Arrays are assigned left-to-right according to their type.
// bit [15:0]$[4] implicitly has type [15:0]$[0:3], so
//   out[0] = D, out[1] = C, out[2] = B, out[3] = A
// CHECK-LABEL: @array_assign_constant
// CHECK-DAG: %[[X0:.*]] = moore.constant 10
// CHECK-DAG: %[[X1:.*]] = moore.constant 11
// CHECK-DAG: %[[X2:.*]] = moore.constant 12
// CHECK-DAG: %[[X3:.*]] = moore.constant 13
// CHECK: moore.array_create %[[X3]], %[[X2]], %[[X1]], %[[X0]]
module array_assign_constant
    (input wire [1 : 0] addr,
     output wire [15 : 0] out);
  localparam bit [15 : 0] kArr[4] = '{16'hD, 16'hC, 16'hB, 16'hA};

  assign out = kArr[addr];
endmodule

// Regression test based on the reproducer from revert PR #10478.
// Verifies that unpacked array literals [0:3] are not reversed by
// ensureDescendingOrder and work correctly with 3 - index dynamic lookup.
// CHECK-LABEL: @repro
// CHECK-DAG: %[[C_MINUS1:.*]] = moore.constant -1 : l2
// CHECK-DAG: %[[C888:.*]] = moore.constant 888 : l16
// CHECK-DAG: %[[C726:.*]] = moore.constant 726 : l16
// CHECK-DAG: %[[C628:.*]] = moore.constant 628 : l16
// CHECK-DAG: %[[C513:.*]] = moore.constant 513 : l16
// CHECK: %[[ARR:.*]] = moore.array_create %[[C513]], %[[C628]], %[[C726]], %[[C888]]
// CHECK: %[[SUB:.*]] = moore.sub %[[C_MINUS1]], %index : l2
// CHECK: %[[EXT:.*]] = moore.dyn_extract %{{.*}} from %[[SUB]]
// CHECK: moore.output %[[EXT]]
module repro(
    input  wire [1:0] index,
    output wire [15:0] out
  );
    wire [15:0] lut [0:3] = '{ 16'h0201, 16'h0274, 16'h02d6, 16'h0378 };
    assign out = lut[index];
endmodule
