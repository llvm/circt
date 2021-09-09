// RUN: circt-opt %s -verify-diagnostics | circt-opt -verify-diagnostics | FileCheck %s

hw.module.extern @fooMod () -> (%x: i32)

// CHECK-LABEL: hw.module @top
hw.module @top () {
  msft.instance "foo" @fooMod () : () -> (i32)
  // CHECK: %foo.x = msft.instance "foo" @fooMod() : () -> i32
}
