// RUN: circt-opt -rtl-legalize-names %s

// expected-error @+1 {{'rtl.module.extern' op with invalid name "parameter"}}
rtl.module.extern @parameter ()
