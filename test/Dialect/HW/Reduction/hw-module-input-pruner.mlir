// UNSUPPORTED: system-windows
//   See https://github.com/llvm/circt/issues/4129
// RUN: circt-reduce %s --test /usr/bin/env --test-arg grep --test-arg -q --test-arg "hw.output %arg" --keep-best=0 --include hw-module-input-pruner | FileCheck %s

// CHECK-LABEL: hw.module @Foo(in %arg0 : i32, out out0 : i32)
hw.module @Foo(in %arg0 : i32, in %arg1 : i32, in %arg2 : i32, out out0 : i32) {
  // CHECK-NEXT: hw.output %arg0 : i32
  hw.output %arg0 : i32
}
