// RUN: circt-opt -pass-pipeline='builtin.module(firrtl.circuit(firrtl-check-recursive-instantiation))' %s | FileCheck %s

// CHECK-LABEL: firrtl.circuit "NoLoop"
firrtl.circuit "NoLoop" {
  firrtl.module @NoLoop() { }
}
