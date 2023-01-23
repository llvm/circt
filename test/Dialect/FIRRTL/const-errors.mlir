// RUN: circt-opt %s -split-input-file -verify-diagnostics

firrtl.circuit "test" {
// expected-error @+1 {{'const' can only be specified once the outermost 'const' type}}
firrtl.module @test(in %a: !firrtl.const.const.uint<1>) {}
}

// -----

firrtl.circuit "test" {
// expected-error @+1 {{'const' can only be specified once the outermost 'const' type}}
firrtl.module @test(in %a: !firrtl.const.bundle<a: const.uint<1>>) {}
}

// -----

firrtl.circuit "test" {
// expected-error @+1 {{'const' can only be specified once the outermost 'const' type}}
firrtl.module @test(in %a: !firrtl.const.vector<const.uint<1>, 2>) {}
}
