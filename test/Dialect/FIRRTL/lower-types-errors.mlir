// RUN: circt-opt -firrtl-lower-types -verify-diagnostics --split-input-file %s

module  {
  firrtl.circuit "top_mod" {
    firrtl.module @top_mod(out %result: !firrtl.uint<8>, in %addr: !firrtl.uint<8>, in %vec0: !firrtl.vector<uint<8>, 4>) {
      %0 = firrtl.subaccess %vec0[%addr] : !firrtl.vector<uint<8>, 4>, !firrtl.uint<8>
      // expected-error @-1 {{SubaccessOp not handled.}}
      firrtl.connect %result, %0 :!firrtl.uint<8>, !firrtl.uint<8>
    }
  }
}

// -----
