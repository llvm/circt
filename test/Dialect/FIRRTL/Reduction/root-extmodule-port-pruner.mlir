// UNSUPPORTED: system-windows
//   See https://github.com/llvm/circt/issues/4129
// RUN: circt-reduce %s --test /usr/bin/env --test-arg true --keep-best=0 --include root-extmodule-port-pruner | FileCheck %s

// Test removing all ports from the root extmodule
// CHECK-LABEL: firrtl.circuit "RootExt"
firrtl.circuit "RootExt" {
  // CHECK-LABEL: firrtl.extmodule @RootExt()
  // CHECK-NOT: in a
  // CHECK-NOT: out b
  // CHECK-NOT: in c
  firrtl.extmodule @RootExt(in a: !firrtl.uint<8>, out b: !firrtl.uint<16>, in c: !firrtl.uint<1>)
}

