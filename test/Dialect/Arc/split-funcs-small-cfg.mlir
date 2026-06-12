// RUN: circt-opt %s --arc-split-funcs=split-bound=3 | FileCheck %s

// CHECK-LABEL: func.func @SmallCfg(
// CHECK-NOT: @SmallCfg_split_func
func.func @SmallCfg(%arg0: i1) {
  cf.cond_br %arg0, ^bb1, ^bb1
^bb1:
  return
}
