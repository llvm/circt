// RUN: circt-opt %s -verify-diagnostics | circt-opt -verify-diagnostics
// RUN: circt-opt %s --msft-lower-instances | FileCheck %s

msft.instance.dynamic [#hw.innerNameRef<@deeper::@branch>, #hw.innerNameRef<@shallow::@leaf>, #hw.innerNameRef<@leaf::@module>] {
  msft.pd.location M20K x: 15 y: 9 n: 3 path: "memBank2"
}

msft.instance.dynamic [#hw.innerNameRef<@shallow::@leaf>, #hw.innerNameRef<@leaf::@module>] {
  msft.pd.location M20K x: 8 y: 19 n: 1 path: "memBank2"
}

msft.instance.dynamic [#hw.innerNameRef<@reg::@reg>] {
  msft.pd.location @ref4 FF x: 0 y: 0 n: 0
}

hw.module.extern @Foo()

// CHECK-LABEL: msft.module @leaf
// LOWER-LABEL: hw.module @leaf
msft.module @leaf {} () -> () {
  // CHECK: msft.instance @module @Foo()
  // LOWER: hw.instance "module" sym @module @Foo() -> ()
  // LOWER-NOT: #msft.switch.inst
  msft.instance @module @Foo() : () -> ()
  // LOWER{LITERAL}: sv.verbatim "proc {{0}}_config
  // LOWER{LITERAL}: sv.verbatim "proc {{0}}_config
  // LOWER{LITERAL}: sv.verbatim "proc {{0}}_config
  msft.output
}

// TCL: Foo module_0

// TCL-NOT: proc leaf_config

// TCL-LABEL: proc shallow_config
msft.module @shallow {} () -> () {
  msft.instance @leaf @leaf() : () -> ()
  // TCL: set_location_assignment M20K_X8_Y19_N1 -to $parent|leaf|module_0|memBank2
  msft.output
}

// TCL-LABEL: proc deeper_config
msft.module @deeper {} () -> () {
  msft.instance @branch @shallow()  : () -> ()
  msft.instance @leaf @leaf() : () -> ()
  // TCL: set_location_assignment M20K_X15_Y9_N3 -to $parent|branch|leaf|module_0|memBank2
  msft.output
}

// TCL-LABEL: proc regions_config
msft.module @regions {} () -> () {
  msft.instance @module @Foo() : () -> ()
  // TCL: set_instance_assignment -name PLACE_REGION "X0 Y0 X10 Y10;X20 Y20 X30 Y30" -to $parent|module_0
  // TCL: set_instance_assignment -name RESERVE_PLACE_REGION OFF -to $parent|module_0
  // TCL: set_instance_assignment -name CORE_ONLY_PLACE_REGION ON -to $parent|module_0
  // TCL: set_instance_assignment -name REGION_NAME region1 -to $parent|module_0
  msft.output
}

// TCL-LABEL: proc reg_0_config
msft.module @reg {} (%input : i8, %clk : i1) -> () {
  %reg = seq.compreg sym @reg %input, %clk  : i8
  // TCL: set_location_assignment FF_X0_Y0_N0 -to $parent|reg_1
  msft.output
}
