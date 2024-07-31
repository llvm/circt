// RUN: circt-opt %s --arc-split-funcs=split-bound=2 --verify-diagnostics

// expected-error @+1 {{Regions with multiple blocks are not supported.}}
func.func @Simple(%arg1: i4, %arg2: i4) -> (i4) {
    %0 = comb.add %arg1, %arg2 : i4
    %1 = comb.xor %arg1, %arg2 : i4
    cf.br ^bb2
^bb2:
    %2 = comb.and %arg1, %arg2 : i4
    return %2 : i4
}
