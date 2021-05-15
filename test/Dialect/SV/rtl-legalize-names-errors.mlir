// RUN: circt-opt -rtl-legalize-names %s

// expected-error @+1 {{'hw.module.extern' op with invalid name "parameter"}}
hw.module.extern @parameter ()
