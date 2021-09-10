// RUN: circt-opt %s -verify-diagnostics | circt-opt -verify-diagnostics | FileCheck %s
// RUN: circt-opt %s --msft-lower-to-hw -verify-diagnostics | circt-opt -verify-diagnostics | FileCheck %s --check-prefix=HWLOW

hw.module.extern @fooMod () -> (%x: i32)

// CHECK-LABEL: hw.module @top
// HWLOW-LABEL: hw.module @top
hw.module @top () {
  msft.instance "foo" @fooMod () : () -> (i32)
  // CHECK: %foo.x = msft.instance "foo" @fooMod() : () -> i32
  // HWLOW: %foo.x = hw.instance "foo" @fooMod() : () -> i32
}
