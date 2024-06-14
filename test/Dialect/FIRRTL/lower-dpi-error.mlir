// RUN: circt-opt -firrtl-lower-dpi %s -verify-diagnostics --split-input-file

// CHECK-LABEL: firrtl.circuit "DPI" {
firrtl.circuit "DPI" {
  firrtl.module @DPI(in %in_0: !firrtl.uint<8>, in %in_1: !firrtl.uint<16>) attributes {convention = #firrtl<convention scalarized>} {
   // expected-error @below {{firrtl.int.dpi.call' op DPI function "foo" input types don't match}}
   firrtl.int.dpi.call "foo"(%in_0) : (!firrtl.uint<8>) -> ()
   // expected-note @below {{mismatched caller is here}}
   firrtl.int.dpi.call "foo"(%in_1) : (!firrtl.uint<16>) -> ()
  }
}

// -----

// CHECK-LABEL: firrtl.circuit "DPI" {
firrtl.circuit "DPI" {
  firrtl.module @DPI(in %in_0: !firrtl.uint<8>) attributes {convention = #firrtl<convention scalarized>} {
   // expected-error @below {{firrtl.int.dpi.call' op DPI function "foo" output types don't match}}
   %0 = firrtl.int.dpi.call "foo"(%in_0) : (!firrtl.uint<8>) -> (!firrtl.uint<16>)
   // expected-note @below {{mismatched caller is here}}
   %1 = firrtl.int.dpi.call "foo"(%in_0) : (!firrtl.uint<8>) -> (!firrtl.uint<8>)
  }
}

