// RUN: circt-opt %s -rtl-declare-typedefs -verify-diagnostics

// expected-error @+1 {{'func' op redefining type definition for sv.typedef}}
func @test(%arg0: !rtl.typealias<foo, i1>, %arg1: !rtl.typealias<foo, i2>) {
  return
}
