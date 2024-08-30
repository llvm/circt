// RUN: circt-opt -firrtl-lower-layers -split-input-file -verify-diagnostics %s

firrtl.circuit "NonPassiveSubaccess" {
  firrtl.layer @A bind {}
  firrtl.module @NonPassiveSubaccess(
    in %a: !firrtl.vector<bundle<a: uint<1>, b flip: uint<1>>, 2>,
    in %b: !firrtl.uint<1>
  ) {
    // expected-note @below {{the layerblock is defined here}}
    firrtl.layerblock @A {
      %n = firrtl.node %b : !firrtl.uint<1>
      // expected-error @below {{'firrtl.subaccess' op has a non-passive operand and captures a value defined outside its enclosing bind-convention layerblock}}
      %0 = firrtl.subaccess %a[%b] : !firrtl.vector<bundle<a: uint<1>, b flip: uint<1>>, 2>, !firrtl.uint<1>
    }
  }
}

// -----

firrtl.circuit "RWProbeCantMove" {
  firrtl.layer @A bind { }
  firrtl.module @RWProbeCantMove() attributes {layers = [@A]} {
    %z = firrtl.constant 0 : !firrtl.uint<5>
    // expected-note @below {{rwprobe target outside of bind layer}}
    %w = firrtl.node sym @sym %z : !firrtl.uint<5>
    firrtl.layerblock @A {
      // expected-error @below {{rwprobe capture not supported with bind convention layer}}
      %rw = firrtl.ref.rwprobe <@RWProbeCantMove::@sym> : !firrtl.rwprobe<uint<5>>
    }
  }
}

