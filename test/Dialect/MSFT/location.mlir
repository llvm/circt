// RUN: circt-opt %s -verify-diagnostics | circt-opt -verify-diagnostics | FileCheck %s
// RUN: circt-opt %s --lower-msft-to-hw=tops=shallow,deeper,regions | FileCheck %s --check-prefix=LOWER
// RUN: circt-opt %s --lower-msft-to-hw=tops=shallow,deeper,regions --export-verilog | FileCheck %s --check-prefix=TCL

hw.module.extern @Foo()

// CHECK-LABEL: msft.module @leaf
// LOWER-LABEL: hw.module @leaf
msft.module @leaf {} () -> () {
  // CHECK: msft.instance @module @Foo()
  // LOWER: hw.instance "module" sym @module @Foo() -> ()
  // LOWER-NOT: #msft.switch.inst
  msft.instance @module @Foo() {
    "loc:memBank2" = #msft.switch.inst< @shallow["leaf"]=#msft.physloc<M20K, 8, 19, 1>,
                                        @deeper["branch","leaf"]=#msft.physloc<M20K, 15, 9, 3> >
  } : () -> ()
  // LOWER: sv.verbatim "proc shallow_config
  // LOWER: sv.verbatim "proc deeper_config
  // LOWER: sv.verbatim "proc regions_config
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
  msft.instance @branch @shallow() : () -> ()
  msft.instance @leaf @leaf() : () -> ()
  // TCL: set_location_assignment M20K_X15_Y9_N3 -to $parent|branch|leaf|module_0|memBank2
  msft.output
}

msft.physical_region @region1, [
  #msft.physical_bounds<x: [0, 10], y: [0, 10]>,
  #msft.physical_bounds<x: [20, 30], y: [20, 30]>]

// TCL-LABEL: proc regions_config
msft.module @regions {} () -> () {
  msft.instance @module @Foo() {
    "msft:loc" = #msft.switch.inst<@regions[]=#msft.physical_region_ref<@region1>>
  } : () -> ()
  // TCL: set_instance_assignment -name PLACE_REGION "X0 Y0 X10 Y10;X20 Y20 X30 Y30" -to $parent|module_0
  // TCL: set_instance_assignment -name RESERVE_PLACE_REGION OFF -to $parent|module_0
  // TCL: set_instance_assignment -name CORE_ONLY_PLACE_REGION ON -to $parent|module_0
  // TCL: set_instance_assignment -name REGION_NAME region1 -to $parent|module_0
  msft.output
}
