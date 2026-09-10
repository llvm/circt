// UNSUPPORTED: system-windows
//   See https://github.com/llvm/circt/issues/4129
// RUN: circt-reduce %s --test /usr/bin/env --test-arg grep --test-arg -q --test-arg "dbg.variable" --keep-best=0 --include hw-module-input-pruner | FileCheck %s

// CHECK-LABEL: hw.module @Foo(in %arg1 : i32, out out0 : i32)
hw.module @Foo(in %arg0 : i32, in %arg1 : i32, in %arg2 : i32, out out0 : i32) {
  // CHECK-NEXT: hw.output %arg1 : i32
  hw.output %arg1 : i32
}

// CHECK-LABEL: hw.module @Bar(in %arg1 : i32)
hw.module @Bar(in %arg0 : i32, in %arg1 : i32, in %arg2 : i32) {
  // CHECK-NEXT: [[TMP:%.+]] = hw.instance "foo" @Foo(arg1: %arg1: i32) -> (out0: i32)
  %0 = hw.instance "foo" @Foo(arg0: %arg0: i32, arg1: %arg1: i32, arg2: %arg2: i32) -> (out0: i32)
  // CHECK-NEXT: dbg.variable "x", [[TMP]]
  dbg.variable "x", %0 : i32
}
