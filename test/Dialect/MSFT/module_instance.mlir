// RUN: circt-opt %s -verify-diagnostics | circt-opt -verify-diagnostics | FileCheck %s

hw.module.extern @fooMod ()

// CHECK-LABEL: hw.module @top
hw.module @top () {
  msft.instance "foo" @fooMod () : () -> ()
  // CHECK: msft.instance "foo" @fooMod() : () -> ()
}
