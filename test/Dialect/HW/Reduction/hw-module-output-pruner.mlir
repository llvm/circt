// UNSUPPORTED: system-windows
//   See https://github.com/llvm/circt/issues/4129
// RUN: circt-reduce %s --test /usr/bin/env --test-arg grep --test-arg -q --test-arg "hw.output %arg" --keep-best=0 --include hw-module-output-pruner-back | FileCheck %s --check-prefix=CHECK-BACK
// RUN: circt-reduce %s --test /usr/bin/env --test-arg grep --test-arg -q --test-arg "hw.output %arg" --keep-best=0 --include hw-module-output-pruner-front | FileCheck %s --check-prefix=CHECK-FRONT

// CHECK-FRONT-LABEL: hw.module @Foo(in %arg0 : i32, in %arg1 : i32, in %arg2 : i32, out out2 : i32)
// CHECK-BACK-LABEL: hw.module @Foo(in %arg0 : i32, in %arg1 : i32, in %arg2 : i32, out out0 : i32)
hw.module @Foo(in %arg0 : i32, in %arg1 : i32, in %arg2 : i32, out out0 : i32, out out1 : i32, out out2 : i32) {
  // CHECK-FRONT-NEXT: hw.output %arg2 : i32
  // CHECK-BACK-NEXT: hw.output %arg0 : i32
  hw.output %arg0, %arg1, %arg2 : i32, i32, i32
}
