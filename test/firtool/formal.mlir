// RUN: firtool %s --verilog | FileCheck %s

// Sanity check to ensure that formal unit tests are lowered to top-level
// modules.

// CHECK: module Foo()
// CHECK: module Bar()

verif.formal @Foo {}
verif.formal @Bar {}
