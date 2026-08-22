// RUN: circt-verilog --ir-moore %s | FileCheck %s --check-prefix=MOORE
// REQUIRES: slang

module M(
    input  logic [7:0]        in1,
    input  logic signed [7:0] in2,
    output logic signed [8:0] mux_out
);
  logic [7:0] unsigned_src;
  logic signed [7:0] signed_src;

  always_comb signed_src = in2 >>> 1;
  always_comb unsigned_src = in1 >> 1;
  always_comb mux_out = signed_src + unsigned_src;
endmodule

// MOORE-LABEL: moore.module @M(
// MOORE: moore.procedure always_comb {
// MOORE:   %[[SIGNED_READ:.+]] = moore.read %signed_src : <l8>
// MOORE:   %[[SIGNED_ZEXT:.+]] = moore.zext %[[SIGNED_READ]] : l8 -> l9
// MOORE-NOT: moore.sext
// MOORE:   %[[UNSIGNED_READ:.+]] = moore.read %unsigned_src : <l8>
// MOORE:   %[[UNSIGNED_ZEXT:.+]] = moore.zext %[[UNSIGNED_READ]] : l8 -> l9
// MOORE:   %[[ADD:.+]] = moore.add %[[SIGNED_ZEXT]], %[[UNSIGNED_ZEXT]] : l9