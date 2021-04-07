// RUN: circt-translate -export-verilog -verify-diagnostics --split-input-file %s

// expected-error @+1 {{value has an unsupported verilog type 'vector<3xi1>'}}
rtl.module @A(%a: vector<3 x i1>) -> () { }
