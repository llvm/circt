// RUN: circt-opt %s -verify-diagnostics | circt-opt -verify-diagnostics | FileCheck %s
// RUN: circt-opt %s --lower-seq-to-sv --msft-export-tcl=tops=shallow,deeper,regions,reg | FileCheck %s --check-prefix=LOWER
// RUN: circt-opt %s --lower-seq-to-sv --msft-export-tcl=tops=shallow,deeper,regions,reg --export-verilog | FileCheck %s --check-prefix=TCL

hw.hierpath @ref1 [@deeper::@branch, @shallow::@leaf, @leaf::@module]
msft.pd.location @ref1 M20K x: 15 y: 9 n: 3 path: "|memBank2"

hw.hierpath @ref2 [@shallow::@leaf, @leaf::@module]
msft.pd.location @ref2 M20K x: 8 y: 19 n: 1 path: "|memBank2"

hw.hierpath @ref3 [@regions::@module]
msft.pd.physregion @ref3 @region1 path: "baz"

hw.hierpath @ref4 [@reg::@reg]
msft.pd.location @ref4 FF x: 0 y: 0 n: 0

hw.module.extern @Foo()

// CHECK-LABEL: hw.module @leaf
// LOWER-LABEL: hw.module @leaf
hw.module @leaf() {
  // CHECK: hw.instance "module" sym @module @Foo()
  // LOWER: hw.instance "module" sym @module @Foo() -> ()
  // LOWER-NOT: #msft.switch.inst
  hw.instance "module" sym @module @Foo() -> ()
  // LOWER{LITERAL}: sv.verbatim "proc {{0}}_config
  // LOWER{LITERAL}: sv.verbatim "proc {{0}}_config
  // LOWER{LITERAL}: sv.verbatim "proc {{0}}_config
}

// TCL: Foo module_0

// TCL-NOT: proc leaf_config

// TCL-LABEL: proc shallow_config
hw.module @shallow() {
  hw.instance "leaf" sym @leaf @leaf() -> ()
  // TCL: set_location_assignment M20K_X8_Y19_N1 -to $parent|leaf|module_0|memBank2
}

// TCL-LABEL: proc deeper_config
hw.module @deeper() {
  hw.instance "branch" sym @branch @shallow() -> ()
  hw.instance "leaf" sym @leaf @leaf() -> ()
  // TCL: set_location_assignment M20K_X15_Y9_N3 -to $parent|branch|leaf|module_0|memBank2
}

msft.physical_region @region1, [
  #msft.physical_bounds<x: [0, 10], y: [0, 10]>,
  #msft.physical_bounds<x: [20, 30], y: [20, 30]>]

// TCL-LABEL: proc regions_config
hw.module @regions() {
  hw.instance "module" sym @module @Foo() -> ()
  // TCL: set_instance_assignment -name PLACE_REGION "X0 Y0 X10 Y10;X20 Y20 X30 Y30" -to $parent|module_0
  // TCL: set_instance_assignment -name RESERVE_PLACE_REGION OFF -to $parent|module_0
  // TCL: set_instance_assignment -name CORE_ONLY_PLACE_REGION ON -to $parent|module_0
  // TCL: set_instance_assignment -name REGION_NAME region1 -to $parent|module_0
}

// TCL-LABEL: proc reg_0_config
hw.module @reg(in %input : i8, in %clk : !seq.clock) {
  %reg = seq.compreg sym @reg %input, %clk : i8
  // TCL: set_location_assignment FF_X0_Y0_N0 -to $parent|reg_0
}
