// RUN: circt-opt %s --arc-split-funcs=split-bound=2 | FileCheck %s

// SplitFuncs can only split a single-block region. A function with multiple
// blocks is left untouched rather than failing the pass.
// CHECK-LABEL: func.func @Simple(
// CHECK-NOT: @Simple_split_func
func.func @Simple(%arg1: i4, %arg2: i4) -> (i4) {
    %0 = comb.add %arg1, %arg2 : i4
    %1 = comb.xor %arg1, %arg2 : i4
    cf.br ^bb2
^bb2:
    %2 = comb.and %arg1, %arg2 : i4
    return %2 : i4
}
