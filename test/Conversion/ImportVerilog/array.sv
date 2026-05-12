// RUN: circt-verilog --ir-moore %s | FileCheck %s
// REQUIRES: slang

// Internal issue in Slang v3 about jump depending on uninitialised value.
// UNSUPPORTED: valgrind

//===---------------------------------------------------------------------===//
// Array element indexing
//===---------------------------------------------------------------------===//
// Array element indexing should return the same value regardless of whether
// the array is packed or unpacked, or reverse-ordered. array[X] should return
// the X'th value.
//
// hw.array's indices are always ordered such that index 0 means the least
// significant element. Therefore regardless of the SV type ([3:0] vs [0:3] vs
// $[4]) we should always plant an array_get of the same index.

// CHECK-LABEL: @array_get_dyn_packed
// CHECK: %[[X:.*]] = moore.dyn_extract %arg0 from %idx
// CHECK-NEXT: moore.output %[[X]]
module array_get_dyn_packed(
  input wire [3:0][15:0] arg0,
  input wire [1:0]       idx,
  output wire [15:0]	 elem
);
   assign elem = arg0[idx];
endmodule

// CHECK-LABEL: @array_get_dyn_packed_rev
// CHECK: %[[C:.*]] = moore.constant -1
// CHECK: %[[S:.*]] = moore.sub %[[C]], %idx
// CHECK: %[[X:.*]] = moore.dyn_extract %arg0 from %[[S]]
// CHECK-NEXT: moore.output %[[X]]
module array_get_dyn_packed_rev(
  input wire [0:3][15:0] arg0,
  input wire [1:0]       idx,
  output wire [15:0]	 elem
);
   assign elem = arg0[idx];
endmodule

// CHECK-LABEL: @array_get_dyn_unpacked
// CHECK: %[[C:.*]] = moore.constant -1
// CHECK: %[[S:.*]] = moore.sub %[[C]], %idx
// CHECK: %[[X:.*]] = moore.dyn_extract %arg0 from %[[S]]
// CHECK-NEXT: moore.output %[[X]]
module array_get_dyn_unpacked(
  input wire [15:0] arg0 [4],
  input wire [1:0]       idx,
  output wire [15:0]	 elem
);
   assign elem = arg0[idx];
endmodule

// CHECK-LABEL: @array_get_dyn_unpacked_rev
// CHECK: %[[X:.*]] = moore.dyn_extract %arg0 from %idx
// CHECK-NEXT: moore.output %[[X]]
module array_get_dyn_unpacked_rev(
  input wire [15:0] arg0 [3:0],
  input wire [1:0]       idx,
  output wire [15:0]	 elem
);
   assign elem = arg0[idx];
endmodule

// CHECK-LABEL: @array_get_static_packed
// CHECK: %[[X:.*]] = moore.extract %arg0 from 1
// CHECK-NEXT: moore.output %[[X]]
module array_get_static_packed(
  input wire [3:0][15:0] arg0,
  output wire [15:0]	 elem
);
   assign elem = arg0[1];
endmodule

// CHECK-LABEL: @array_get_static_packed_rev
// CHECK: %[[X:.*]] = moore.extract %arg0 from 2
// CHECK-NEXT: moore.output %[[X]]
module array_get_static_packed_rev(
  input wire [0:3][15:0] arg0,
  output wire [15:0]	 elem
);
   assign elem = arg0[1];
endmodule

// CHECK-LABEL: @array_get_static_unpacked
// CHECK: %[[X:.*]] = moore.extract %arg0 from 2
// CHECK-NEXT: moore.output %[[X]]
module array_get_static_unpacked(
  input wire [15:0] arg0 [4],
  output wire [15:0]	 elem
);
   assign elem = arg0[1];
endmodule

// CHECK-LABEL: @array_get_static_unpacked_rev
// CHECK: %[[X:.*]] = moore.extract %arg0 from 1
// CHECK-NEXT: moore.output %[[X]]
module array_get_static_unpacked_rev(
  input wire [15:0] arg0 [3:0],
  output wire [15:0]	 elem
);
   assign elem = arg0[1];
endmodule

// CHECK-LABEL: @array_get_static_unpacked_rev
// CHECK: %[[X:.*]] = moore.extract %arg0 from 2
module array_get_static_unpacked_rev_rev(
  input wire [15:0] arg0 [0:3],
  output wire [15:0]	 elem
);
   assign elem = arg0[1];
endmodule

// CHECK-LABEL: @array_slice_dyn_packed
// CHECK: %[[X:.*]] = moore.dyn_extract %arg0 from %idx
// CHECK-NEXT: moore.output %[[X]]
module array_slice_dyn_packed(
  input wire [7:0][15:0] arg0,
  input wire [2:0]       idx,
  output wire [3:0][15:0] slice
);
  assign slice = arg0[idx+:4];
endmodule

// CHECK-LABEL: @array_slice_dyn_packed_rev
// CHECK: %[[S:.*]] = moore.sub
// CHECK: %[[X:.*]] = moore.dyn_extract %arg0 from %[[S]]
// CHECK-NEXT: moore.output %[[X]]
module array_slice_dyn_packed_rev(
  input wire [0:7][15:0] arg0,
  input wire [2:0]       idx,
  output wire [3:0][15:0] slice
);
  assign slice = arg0[idx+:4];
endmodule

// CHECK-LABEL: @array_slice_dyn_unpacked
// CHECK: %[[S:.*]] = moore.sub
// CHECK: %[[X:.*]] = moore.dyn_extract %arg0 from %[[S]]
// CHECK-NEXT: moore.output %[[X]]
module array_slice_dyn_unpacked(
  input wire [15:0] arg0 [8],
  input wire [2:0]       idx,
  output wire [15:0] slice [4]
);
  assign slice = arg0[idx+:4];
endmodule

// CHECK-LABEL: @array_slice_static_packed
// CHECK: %[[X:.*]] = moore.extract %arg0 from 1
// CHECK-NEXT: moore.output %[[X]]
module array_slice_static_packed(
  input wire [7:0][15:0] arg0,
  output wire [3:0][15:0] slice
);
  assign slice = arg0[1+:4];
endmodule

// CHECK-LABEL: @array_slice_static_unpacked
// CHECK: %[[X:.*]] = moore.extract %arg0 from 3
// CHECK-NEXT: moore.output %[[X]]
module array_slice_static_unpacked(
  input wire [15:0] arg0 [8],
  output wire [15:0] slice [4]
);
  assign slice = arg0[1+:4];
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
module array_assign_packed(
  input wire [15:0] a, b, c, d,
  output wire [3:0][15:0] out
);
  assign out = '{d, c, b, a};
endmodule

// CHECK-LABEL: @array_assign_unpacked
// CHECK: moore.array_create %a, %b, %c, %d
module array_assign_unpacked(
  input wire [15:0] a, b, c, d,
  output wire [15:0] out [4]
);
  assign out = '{d, c, b, a};
endmodule

// CHECK-LABEL: @array_assign_rev_packed
// CHECK: moore.array_create %a, %b, %c, %d
module array_assign_rev_packed(
  input wire [15:0] a, b, c, d,
  output wire [0:3][15:0] out
);
  assign out = '{d, c, b, a};
endmodule

// CHECK-LABEL: @array_assign_rev_unpacked
// CHECK: moore.array_create %d, %c, %b, %a
module array_assign_rev_unpacked(
  input wire [15:0] a, b, c, d,
  output wire [15:0] out [3:0]
);
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
// CHECK: moore.array_create %[[X0]], %[[X1]], %[[X2]], %[[X3]]
module array_assign_constant(
  input wire [1:0] addr,
  output wire [15:0] out
);
  localparam bit [15:0] kArr[4] = '{16'hD, 16'hC, 16'hB, 16'hA};

  assign out = kArr[addr];
endmodule

// As above, but using the explicit [0:3] syntax.
// CHECK-LABEL: @array_assign_constant_explicit
// CHECK-DAG: %[[X0:.*]] = moore.constant 10
// CHECK-DAG: %[[X1:.*]] = moore.constant 11
// CHECK-DAG: %[[X2:.*]] = moore.constant 12
// CHECK-DAG: %[[X3:.*]] = moore.constant 13
// CHECK: moore.array_create %[[X0]], %[[X1]], %[[X2]], %[[X3]]
module array_assign_constant_explicit(
  input wire [1:0] addr,
  output wire [15:0] out
);
  localparam bit [15:0] kArr[0:3] = '{16'hD, 16'hC, 16'hB, 16'hA};

  assign out = kArr[addr];
endmodule

// Explicitly using the reversed syntax [3:0]. Again, assign left-to-right,
// so out[3] = D, out[2] = C, out[1] = B, out[0] = A.
// CHECK-LABEL: @array_assign_constant_explicit_rev
// CHECK-DAG: %[[X0:.*]] = moore.constant 10
// CHECK-DAG: %[[X1:.*]] = moore.constant 11
// CHECK-DAG: %[[X2:.*]] = moore.constant 12
// CHECK-DAG: %[[X3:.*]] = moore.constant 13
// CHECK: moore.array_create %[[X3]], %[[X2]], %[[X1]], %[[X0]]
module array_assign_constant_explicit_rev(
  input wire [1:0] addr,
  output wire [15:0] out
);
  localparam bit [15:0] kArr[3:0] = '{16'hD, 16'hC, 16'hB, 16'hA};

  assign out = kArr[addr];
endmodule

// The same rules apply for integers; assignment proceeds from left-to-right,
// so out[1] = 0, out[0] = 1.
// CHECK-LABEL: @int_assign_constant
// CHECK-DAG: %[[X0:.*]] = moore.constant 0 :
// CHECK-DAG: %[[X1:.*]] = moore.constant 1 :
// CHECK: moore.concat %[[X1]], %[[X0]]
module int_assign_constant(
  output wire [1:0] out
);
  assign out = '{0, 1};
endmodule

// CHECK-LABEL: @int_assign_constant_rev
// CHECK-DAG: %[[X0:.*]] = moore.constant 0 :
// CHECK-DAG: %[[X1:.*]] = moore.constant 1 :
// CHECK: moore.concat %[[X0]], %[[X1]]
module int_assign_constant_rev(
  output wire [0:1] out
);
  assign out = '{0, 1};
endmodule

//===----------------------------------------------------------------------===//
// Array concatenation
//===----------------------------------------------------------------------===//
//
// The concatenation operator's operands are always ordered from most to least
// significant:
//  wire [2:0] a = {1, 2, 3}  // a[0] == 3
//  wire [0:2] b = {1, 2, 3}  // b[0] == 3

module array_concat_constant_packed(
  output wire [2:0][3:0] out
);
  assign out = {4'd1, 4'd2, 4'd3};
endmodule

module array_concat_constant_packed_rev(
  output wire [0:2][3:0] out
);
  assign out = {4'd1, 4'd2, 4'd3};
endmodule


// Note that the standard says that this should not compile!
module array_concat_constant_unpacked(
  output wire [3:0] out [3]
);
  assign out = {4'd1, 4'd2, 4'd3};
endmodule
