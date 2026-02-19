// UNSUPPORTED: system-windows
//   See https://github.com/llvm/circt/issues/4129
// RUN: circt-reduce %s --test /usr/bin/env --test-arg true --keep-best=0 --include module-port-pruner | FileCheck %s

// Test removing unused ports from a regular module with instances
firrtl.circuit "ModuleWithInstances" {
  // CHECK-LABEL: firrtl.module @ModuleWithInstances
  firrtl.module @ModuleWithInstances() {
    // CHECK: %sub_b = firrtl.instance sub @Sub
    // CHECK-NOT: %sub_a
    // CHECK-NOT: %sub_c
    %sub_a, %sub_b, %sub_c = firrtl.instance sub @Sub(in a: !firrtl.uint<1>, out b: !firrtl.uint<1>, out c: !firrtl.uint<1>)
    %w = firrtl.wire : !firrtl.uint<1>
    firrtl.connect %w, %sub_b : !firrtl.uint<1>, !firrtl.uint<1>
  }

  // CHECK-LABEL: firrtl.module private @Sub
  // CHECK-NOT: in %a
  // CHECK-SAME: out %b
  // CHECK-NOT: out %c
  firrtl.module private @Sub(in %a: !firrtl.uint<1>, out %b: !firrtl.uint<1>, out %c: !firrtl.uint<1>) {
    %invalid = firrtl.invalidvalue : !firrtl.uint<1>
    firrtl.connect %b, %invalid : !firrtl.uint<1>, !firrtl.uint<1>
    firrtl.connect %c, %invalid : !firrtl.uint<1>, !firrtl.uint<1>
  }
}

// Test removing unused input ports from an uninstantiated module
firrtl.circuit "UninstantiatedModule" {
  firrtl.module @UninstantiatedModule() {
  }

  // CHECK-LABEL: firrtl.module private @Unused
  // CHECK-NOT: in %a
  // CHECK-SAME: out %b
  firrtl.module private @Unused(in %a: !firrtl.uint<1>, out %b: !firrtl.uint<1>) {
    %invalid = firrtl.invalidvalue : !firrtl.uint<1>
    firrtl.connect %b, %invalid : !firrtl.uint<1>, !firrtl.uint<1>
  }
}

