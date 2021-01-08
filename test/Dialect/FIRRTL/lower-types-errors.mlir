// RUN: firtool %s -format=mlir -lower-to-rtl | circt-opt -verify-diagnostics | FileCheck %s --check-prefix=LOWER
// RUN: firtool %s -format=mlir -lower-to-rtl -enable-lower-types | circt-opt -verify-diagnostics | FileCheck %s --check-prefix=LOWERTYPES

firrtl.circuit "Top" {
  firrtl.module @Top(%in : !firrtl.bundle<a: uint<1>, b: uint<1>>,
                     %out : !firrtl.bundle<a: flip<uint<1>>, b: flip<uint<1>>>) {
    firrtl.connect %out, %in : !firrtl.bundle<a: flip<uint<1>>, b: flip<uint<1>>>, !firrtl.bundle<a: uint<1>, b: uint<1>>
  }
}

// LOWER-LABEL: module attributes {firrtl.mainModule = "Top"}
// expected-error: @+1 {{cannot lower this port type to RTL}}

// LOWERTYPES-LABEL: module attributes {firrtl.mainModule = "Top"}
// LOWERTYPES: rtl.output %in_a, %in_b
