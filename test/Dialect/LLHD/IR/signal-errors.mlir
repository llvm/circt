// RUN: circt-opt %s -mlir-print-op-generic -split-input-file -verify-diagnostics

// expected-error @+3 {{failed to verify that type of 'init' and underlying type of 'signal' have to match.}}
hw.module @check_illegal_sig() {
  %cI1 = hw.constant 0 : i1
  %sig1 = "llhd.sig"(%cI1) {name="foo"} : (i1) -> !hw.inout<i32>
}

// -----

// expected-error @+2 {{failed to verify that type of 'result' and underlying type of 'signal' have to match.}}
hw.module @check_illegal_prb(inout %sig : i1) {
  %prb = "llhd.prb"(%sig) {} : (!hw.inout<i1>) -> i32
}

// -----

// expected-error @+4 {{failed to verify that type of 'value' and underlying type of 'signal' have to match.}}
hw.module @check_illegal_drv(inout %sig : i1) {
  %c = hw.constant 0 : i32
  %time = llhd.constant_time #llhd.time<1ns, 0d, 0e>
  "llhd.drv"(%sig, %c, %time) {} : (!hw.inout<i1>, i32, !llhd.time) -> ()
}

// -----

func.func @illegal_sig_parent (%arg0: i1) {
  // expected-error @+1 {{expects parent op to be one of 'hw.module, llhd.process'}}
  %0 = llhd.sig "sig" %arg0 : i1
}
