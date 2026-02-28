// RUN: circt-opt %s --arc-lower-state --verify-diagnostics --split-input-file

hw.module @CombLoop(in %a: i42, out z: i42) {
  // expected-error @below {{'comb.add' op is on a combinational loop}}
  // expected-remark @below {{computing new phase here}}
  %0 = comb.add %a, %1 : i42
  // expected-remark @below {{computing new phase here}}
  %1 = comb.mul %a, %0 : i42
  // expected-remark @below {{computing new phase here}}
  hw.output %0 : i42
}

// -----

// expected-error @+1 {{Failed to remove external module because it is still referenced/instantiated}}
hw.module.extern @myModule()

hw.instance "alligator" @myModule() -> ()
