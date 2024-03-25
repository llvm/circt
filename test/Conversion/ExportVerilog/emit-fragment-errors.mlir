// RUN: circt-opt %s -export-verilog -verify-diagnostics

// expected-error @below {{cannot find referenced fragment @DoesNotExis}}
hw.module @SomeModule(in %in : i32, out out : i32) attributes { "emit.fragments" = [@DoesNotExist] } {
  hw.output %in : i32
}
