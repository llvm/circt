// RUN: circt-opt %s | circt-opt | FileCheck %s

// CHECK-LABEL: func @types
// CHECK-SAME:  (%{{.*}}: !smt.bool, %{{.*}}: !smt.bv<32>)
func.func @types(%arg0: !smt.bool, %arg1: !smt.bv<32>) {
  return
}
