// RUN: circt-opt %s -mlir-print-op-generic -split-input-file -verify-diagnostics

// expected-error @+3 {{failed to verify that type of 'init' and underlying type of 'signal' have to match.}}
llhd.entity @check_illegal_sig () -> () {
  %cI1 = hw.constant 0 : i1
  %sig1 = "llhd.sig"(%cI1) {name="foo"} : (i1) -> !llhd.sig<i32>
}

// -----

// expected-error @+2 {{failed to verify that type of 'result' and underlying type of 'signal' have to match.}}
llhd.entity @check_illegal_prb (%sig : !llhd.sig<i1>) -> () {
  %prb = "llhd.prb"(%sig) {} : (!llhd.sig<i1>) -> i32
}

// -----

// expected-error @+4 {{failed to verify that type of 'value' and underlying type of 'signal' have to match.}}
llhd.entity @check_illegal_drv (%sig : !llhd.sig<i1>) -> () {
  %c = hw.constant 0 : i32
  %time = llhd.constant_time #llhd.time<1ns, 0d, 0e>
  "llhd.drv"(%sig, %c, %time) {} : (!llhd.sig<i1>, i32, !llhd.time) -> ()
}

// -----

// expected-error @+4 {{Redefinition of signal named 'sigI1'!}}
llhd.entity @check_unique_sig_names () -> () {
  %cI1 = hw.constant 0 : i1
  %sig1 = llhd.sig "sigI1" %cI1 : i1
  %sig2 = llhd.sig "sigI1" %cI1 : i1
}
