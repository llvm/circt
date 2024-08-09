// RUN: circt-opt -firrtl-lower-layers -split-input-file -verify-diagnostics %s

firrtl.circuit "RWProbeCantMove" {
  firrtl.layer @A bind { }
  firrtl.module @RWProbeCantMove() attributes {layers = [@A]} {
    %z = firrtl.constant 0 : !firrtl.uint<5>
    %w = firrtl.node sym @sym %z : !firrtl.uint<5>
    firrtl.layerblock @A {
      // expected-error @below {{rwprobe target not moved}}
      %rw = firrtl.ref.rwprobe <@RWProbeCantMove::@sym> : !firrtl.rwprobe<uint<5>>
    }
  }
}

