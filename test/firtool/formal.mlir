// RUN: firtool %s --verilog | FileCheck %s

// Sanity check to ensure that formal unit tests are lowered to top-level
// modules.

// CHECK: module Foo()
verif.formal @Foo {} {
}

// CHECK: module Bar()
verif.formal @Bar {} {
}
