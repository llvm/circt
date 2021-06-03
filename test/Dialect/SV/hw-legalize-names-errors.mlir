// RUN: circt-opt -hw-legalize-names %s

// expected-error @+1 {{'hw.module.extern' op with invalid name "parameter"}}
hw.module.extern @parameter ()

// expected-error @+1 {{'hw.module.extern' op redefining type definition for hw.typedecl}}
hw.module.extern @test(
  %arg0: !hw.typealias<@scope::@foo, i1>,
  %arg1: !hw.typealias<@scope::@foo, i2>)
