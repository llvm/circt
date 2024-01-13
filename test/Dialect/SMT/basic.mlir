// RUN: circt-opt %s | circt-opt | FileCheck %s

// CHECK-LABEL: func @types
// CHECK-SAME:  (%{{.*}}: !smt.bool)
func.func @types(%arg0: !smt.bool) {
  return
}
