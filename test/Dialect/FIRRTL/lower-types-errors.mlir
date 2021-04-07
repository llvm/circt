// RUN: circt-opt -firrtl-lower-types -verify-diagnostics --split-input-file %s

module  {
  firrtl.circuit "top_mod" {
    firrtl.module @top_mod(%result: !firrtl.flip<uint<8>>, %addr: !firrtl.uint<8>, %vec0: !firrtl.vector<uint<8>, 4>) {
      %0 = firrtl.subaccess %vec0[%addr] : !firrtl.vector<uint<8>, 4>, !firrtl.uint<8>
      // expected-error @-1 {{SubaccessOp not handled.}}
      firrtl.connect %result, %0 :!firrtl.flip<uint<8>>, !firrtl.uint<8>
    }
  }
}

// -----
