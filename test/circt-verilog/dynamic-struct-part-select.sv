// clang-format off
// RUN: circt-verilog --ir-moore %s | FileCheck %s --check-prefix=MOORE
// RUN: circt-verilog --ir-llhd %s | FileCheck %s --check-prefix=LLHD
// RUN: circt-verilog --ir-hw %s | FileCheck %s --check-prefix=HW
// RUN: circt-verilog --ir-hw --sroa %s | FileCheck %s --check-prefix=HW
// REQUIRES: slang
// UNSUPPORTED: valgrind

// The original reproducer: the selected value itself is a packed struct.
// MOORE-LABEL: moore.module @lanes
// MOORE: moore.dyn_extract_ref {{.*}} : <struct<{meta: l3, data: l128}>>, i32 -> <l32>
// LLHD-LABEL: hw.module @lanes
// LLHD: llhd.sig.extract {{.*}} : <!hw.struct<meta: i3, data: i128>> -> <i32>
// HW-LABEL: hw.module @lanes
// HW-NOT: llhd.
// HW: hw.output
module lanes(
  input logic [3:0][31:0] narrow_i,
  output struct packed { logic [2:0] meta; logic [127:0] data; } payload_o
);
  always_comb begin
    for (int i = 0; i < 4; i++)
      payload_o[i*32 +: 32] = narrow_i[i];
  end
endmodule

typedef struct packed {
  bit [7:0] hi;
  bit [7:0] lo;
} pair_t;

// The six-bit writes include a field boundary crossing. Preserve the surrounding
// bits of the input after unrolling the loop.
// MOORE-LABEL: moore.module @cross_fields
// LLHD-LABEL: hw.module @cross_fields
// HW-LABEL: hw.module @cross_fields
// HW-DAG: [[MASK1:%.+]] = hw.constant -1009 : i16
// HW-DAG: [[VALUE1:%.+]] = hw.constant 1008 : i16
// HW-DAG: [[MASK2:%.+]] = hw.constant -16129 : i16
// HW-DAG: [[VALUE2:%.+]] = hw.constant 16128 : i16
// HW: [[KEEP1:%.+]] = comb.and %initial_value, [[MASK1]] : i16
// HW-NEXT: [[WRITE1:%.+]] = comb.or [[KEEP1]], [[VALUE1]] : i16
// HW-NEXT: [[KEEP2:%.+]] = comb.and [[WRITE1]], [[MASK2]] : i16
// HW-NEXT: [[WRITE2:%.+]] = comb.or [[KEEP2]], [[VALUE2]] : i16
// HW-NEXT: hw.output [[WRITE2]] : i16
module cross_fields(input bit [15:0] initial_value, output bit [15:0] result);
  pair_t payload;
  always_comb begin
    payload = initial_value;
    for (int i = 0; i < 2; i++)
      payload[i*4 + 4 +: 6] = 6'h3f;
  end
  assign result = payload;
endmodule

typedef struct packed {
  struct packed { bit [3:0] a; bit [3:0] b; } hi;
  bit [1:0][3:0] lo;
} nested_t;

// An input-dependent index into nested packed aggregates.
// MOORE-LABEL: moore.module @nested
// LLHD-LABEL: hw.module @nested
// LLHD: llhd.sig.extract {{.*}} : <!hw.struct<hi: !hw.struct<a: i4, b: i4>, lo: !hw.array<2xi4>>> -> <i6>
// HW-LABEL: hw.module @nested
// HW-NOT: llhd.
// HW: hw.output
module nested(input nested_t initial_value, input bit [3:0] index,
              input bit [5:0] value, output nested_t result);
  always_comb begin
    result = initial_value;
    result[index +: 6] = value;
  end
endmodule

typedef struct packed { bit [2:0] meta; pair_t data; } envelope_t;

// The selected struct may itself be a field of another struct, as in the DMA
// payload that motivated this regression.
// MOORE-LABEL: moore.module @nested_field
// LLHD-LABEL: hw.module @nested_field
// LLHD: llhd.sig.struct_extract {{.*}}["data"]
// LLHD: llhd.sig.extract {{.*}} : <!hw.struct<hi: i8, lo: i8>> -> <i6>
// HW-LABEL: hw.module @nested_field
// HW-NOT: llhd.
// HW: hw.output
module nested_field(input envelope_t initial_value, input bit [3:0] index,
                    input bit [5:0] value, output envelope_t result);
  always_comb begin
    result = initial_value;
    result.data[index +: 6] = value;
  end
endmodule

// Nonblocking writes must continue to update only the selected bits.
// MOORE-LABEL: moore.module @clocked
// LLHD-LABEL: hw.module @clocked
// LLHD: llhd.sig.extract {{.*}} : <!hw.struct<hi: i8, lo: i8>> -> <i6>
// HW-LABEL: hw.module @clocked
// HW: seq.firreg
// HW: hw.output
module clocked(input logic clk, input bit [3:0] index,
               input bit [5:0] value, output pair_t result);
  always_ff @(posedge clk)
    result[index +: 6] <= value;
endmodule
