// RUN: circt-opt %s --convert-llhd-to-llvm --verify-diagnostics --split-input-file

hw.module @root() {
  // expected-error @+1 {{failed to legalize operation 'hw.instance'}}
  hw.instance "inst" @initUsesProbedValue () -> ()
}

// Test: initializer expression of second signal uses probe of another signal
hw.module @initUsesProbedValue() {
  // expected-error @+1 {{failed to legalize operation 'hw.constant'}}
  %0 = hw.constant 0 : i1
  %1 = llhd.sig "sig" %0 : i1
  %2 = llhd.prb %1 : !hw.inout<i1>
  %3 = hw.array_create %2, %2 : i1
  %4 = llhd.sig "sig1" %3 : !hw.array<2xi1>
}

// TODO: add testcase where the init value of llhd.sig comes from a block argument
