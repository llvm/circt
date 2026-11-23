// RUN: circt-opt %s --lower-firrtl-to-hw --verify-diagnostics

firrtl.circuit "BitcastBundle" {
  firrtl.module @BitcastBundle(in %input_0: !firrtl.uint<8>, out %output_0: !firrtl.uint<8>) {
    %io = firrtl.wire : !firrtl.bundle<input_0 flip: uint<8>, output_0: uint<8>>
    %0 = firrtl.subfield %io[input_0] : !firrtl.bundle<input_0 flip: uint<8>, output_0: uint<8>>
    firrtl.connect %0, %input_0 : !firrtl.uint<8>
    %1 = firrtl.subfield %io[output_0] : !firrtl.bundle<input_0 flip: uint<8>, output_0: uint<8>>
    firrtl.connect %output_0, %1 : !firrtl.uint<8>
    %2 = firrtl.subfield %io[output_0] : !firrtl.bundle<input_0 flip: uint<8>, output_0: uint<8>>
    // expected-error @below {{cannot cast input bundle type with elements in different directions '!firrtl.bundle<input_0 flip: uint<8>, output_0: uint<8>>'}}
    %3 = firrtl.bitcast %io : (!firrtl.bundle<input_0 flip: uint<8>, output_0: uint<8>>) -> !firrtl.uint<16>
    %4 = firrtl.bits %3 7 to 0 : (!firrtl.uint<16>) -> !firrtl.uint<8>
    %_GEN_0 = firrtl.node interesting_name %4 : !firrtl.uint<8>
    firrtl.connect %2, %_GEN_0 : !firrtl.uint<8>
  }
}
