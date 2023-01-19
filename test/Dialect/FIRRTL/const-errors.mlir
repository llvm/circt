// RUN: circt-opt %s -split-input-file -verify-diagnostics

firrtl.circuit "test" {
// expected-error @+1 {{analog types cannot be 'const'}}
firrtl.module @test(in %a: !firrtl.const.analog<1>) {}
}

// -----

firrtl.circuit "test" {
// expected-error @+1 {{clock types cannot be 'const'}}
firrtl.module @test(in %a: !firrtl.const.clock) {}
}

// -----

firrtl.circuit "test" {
// expected-error @+1 {{reset types cannot be 'const'}}
firrtl.module @test(in %a: !firrtl.const.reset) {}
}

// -----

firrtl.circuit "test" {
// expected-error @+1 {{asyncreset types cannot be 'const'}}
firrtl.module @test(in %a: !firrtl.const.asyncreset) {}
}

// -----

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
