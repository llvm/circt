// RUN: circt-opt %s -hw-declare-typedecls -verify-diagnostics

// expected-error @+1 {{'func' op redefining type definition for hw.typedecl}}
func @test(
  %arg0: !hw.typealias<@scope::@foo, i1>,
  %arg1: !hw.typealias<@scope::@foo, i2>) {
  return
}
