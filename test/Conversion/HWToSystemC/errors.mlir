// RUN: circt-opt --convert-hw-to-systemc --verify-diagnostics --split-input-file %s

// expected-error @+2 {{module parameters not supported yet}}
// expected-error @+1 {{failed to legalize operation 'hw.module'}}
hw.module @someModule<p1: i42 = 17, p2: i1>() -> () {}

// -----

// expected-error @+2 {{inout arguments not supported yet}}
// expected-error @+1 {{failed to legalize operation 'hw.module'}}
hw.module @someModule(%in0: !hw.inout<i32>) -> () {}
