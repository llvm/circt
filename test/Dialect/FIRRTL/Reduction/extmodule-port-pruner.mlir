// UNSUPPORTED: system-windows
//   See https://github.com/llvm/circt/issues/4129
// RUN: circt-reduce %s --test /usr/bin/env --test-arg true --keep-best=0 --include extmodule-port-pruner | FileCheck %s

// Test removing unused ports from an extmodule with instances
firrtl.circuit "ExtmoduleWithInstances" {
  // CHECK-LABEL: firrtl.module @ExtmoduleWithInstances
  firrtl.module @ExtmoduleWithInstances() {
    // CHECK: %ext_b = firrtl.instance ext @Ext
    // CHECK-NOT: %ext_a
    // CHECK-NOT: %ext_c
    %ext_a, %ext_b, %ext_c = firrtl.instance ext @Ext(in a: !firrtl.uint<1>, out b: !firrtl.uint<1>, out c: !firrtl.uint<1>)
    %w = firrtl.wire : !firrtl.uint<1>
    firrtl.connect %w, %ext_b : !firrtl.uint<1>, !firrtl.uint<1>
  }

  // CHECK-LABEL: firrtl.extmodule private @Ext
  // CHECK-NOT: in a
  // CHECK-SAME: out b
  // CHECK-NOT: out c
  firrtl.extmodule private @Ext(in a: !firrtl.uint<1>, out b: !firrtl.uint<1>, out c: !firrtl.uint<1>)
}

// Uninstantiated extmodules have all ports removed
firrtl.circuit "UninstantiatedExtmodule" {
  firrtl.module @UninstantiatedExtmodule() {
  }

  // CHECK-LABEL: firrtl.extmodule private @UnusedExt
  // CHECK-NOT: in a
  // CHECK-NOT: out b
  firrtl.extmodule private @UnusedExt(in a: !firrtl.uint<1>, out b: !firrtl.uint<1>)
}

