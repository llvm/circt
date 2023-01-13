// RUN: circt-opt --firrtl-undef %s | FileCheck %s

firrtl.circuit "MyModule" {

// Constant op supports different return types.
firrtl.module @Constants() {
  %c0 = firrtl.constant 0 : !firrtl.uint<3>
  %i0 = firrtl.invalidvalue : !firrtl.uint<3>
  firrtl.specialconstant 1 : !firrtl.clock
  firrtl.aggregateconstant [1, 2, 3] : !firrtl.bundle<a: uint<8>, b: uint<5>, c: uint<4>>
  firrtl.aggregateconstant [1, 2, 3] : !firrtl.vector<uint<8>, 3>
  %0 = firrtl.add %c0, %c0 : (!firrtl.uint<3>, !firrtl.uint<3>) -> !firrtl.uint<4>
  %1 = firrtl.add %c0, %i0 : (!firrtl.uint<3>, !firrtl.uint<3>) -> !firrtl.uint<4>
}

firrtl.module @ResetFixesEverything(in %clock: !firrtl.clock, in %reset : !firrtl.uint<1>) {
  %c0 = firrtl.constant 0 : !firrtl.uint<3>
  %r1 = firrtl.regreset %clock, %reset, %c0 : !firrtl.uint<1>, !firrtl.uint<3>, !firrtl.uint<3>
  %r2 = firrtl.reg %clock : !firrtl.uint<3>
  firrtl.strictconnect %r2, %r1 : !firrtl.uint<3>  
}

firrtl.module @ResetFixesNothing(in %clock: !firrtl.clock, in %reset : !firrtl.uint<1>) {
  %c0 = firrtl.invalidvalue : !firrtl.uint<3>
  %r1 = firrtl.regreset %clock, %reset, %c0 : !firrtl.uint<1>, !firrtl.uint<3>, !firrtl.uint<3>
  %r2 = firrtl.reg %clock : !firrtl.uint<3>
  firrtl.strictconnect %r2, %r1 : !firrtl.uint<3>  
}

firrtl.module @MyModule(in %in : !firrtl.uint<8>,
                        in %clock : !firrtl.clock,
                        in %reset : !firrtl.uint<1>,
                        out %out : !firrtl.uint<8>) {
  firrtl.instance iConstants @Constants ()
  firrtl.instance iConstants2 @Constants ()
  %irc, %irr = firrtl.instance iResetFixesEverything @ResetFixesEverything (in clock : !firrtl.clock, in reset : !firrtl.uint<1>)
  %irc2, %irr2 = firrtl.instance iResetFixesNothing @ResetFixesNothing (in clock : !firrtl.clock, in reset : !firrtl.uint<1>)
  firrtl.strictconnect %irr, %reset : !firrtl.uint<1>
  firrtl.strictconnect %irc, %clock : !firrtl.clock
  firrtl.strictconnect %irr2, %reset : !firrtl.uint<1>
  firrtl.strictconnect %irc2, %clock : !firrtl.clock
  %i = firrtl.invalidvalue : !firrtl.uint<8>
  firrtl.strictconnect %out, %i : !firrtl.uint<8>
}

}