// RUN: firtool %s -format=mlir -verilog -verify-diagnostics | FileCheck --allow-empty %s

firrtl.circuit "top" {
  // expected-error @below {{'firrtl.module' op contains a 'chisel3.util.experimental.ForceNameAnnotation' that is not a non-local annotation}}
  // expected-note @below {{the erroneous annotation is '{class = "chisel3.util.experimental.ForceNameAnnotation"}'}}
  // expected-warning @below {{module `top` is empty but cannot be removed because the module is public}}
  firrtl.module @top() attributes {annotations = [{class = "chisel3.util.experimental.ForceNameAnnotation"}]} {
  }
}

// Ensure circuits that error out don't make it to ExportVerilog.

// CHECK-NOT: module top
