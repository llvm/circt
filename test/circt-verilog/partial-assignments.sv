// RUN: circt-verilog %s | FileCheck %s
// REQUIRES: slang
// UNSUPPORTED: valgrind

// The index is a variable whose value is known. The first `llhd-sig2reg` run
// cannot promote `out`, since its drive reads `out` itself through a mask built
// from the not yet promoted `ind`. Once `ind` is promoted, the canonicalizer
// folds the mask into a `comb.concat` and the second run promotes `out`.
// CHECK-LABEL: hw.module @KnownVariableIndex(out out : i6) {
// CHECK-NEXT: [[ONE:%.+]] = hw.constant 1 : i6
// CHECK-NEXT: hw.output [[ONE]] : i6
module KnownVariableIndex(output bit [5:0] out);
  logic ind = 0;
  always @* begin
    out[ind] = 1;
  end
endmodule
