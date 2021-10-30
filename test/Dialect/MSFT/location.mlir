// RUN: circt-opt %s -verify-diagnostics | circt-opt -verify-diagnostics | FileCheck %s
// RUN: circt-translate %s --export-quartus-tcl | FileCheck %s --check-prefix=TCL

hw.module.extern @Foo()

// CHECK-LABEL: hw.module @leaf
hw.module @leaf() {
  // CHECK: hw.instance "foo" @Foo() -> () {"loc:memBank2" = #msft.switch.inst<@shallow["leaf"]=#msft.physloc<M20K, 8, 19, 1>, @deeper["branch","leaf"]=#msft.physloc<M20K, 15, 9, 3>>} 
  hw.instance "foo" @Foo() -> () {
    "loc:memBank2" = #msft.switch.inst< @shallow["leaf"]=#msft.physloc<M20K, 8, 19, 1>,
                                        @deeper["branch","leaf"]=#msft.physloc<M20K, 15, 9, 3> > }
}

// TCL-LABEL: proc shallow_config
hw.module @shallow() {
  hw.instance "leaf" @leaf() -> ()
  // TCL: set_location_assignment M20K_X8_Y19_N1 -to $parent|leaf|foo|memBank2
}

// TCL-LABEL: proc deeper_config
hw.module @deeper() {
  hw.instance "branch" @shallow() -> ()
  hw.instance "leaf" @leaf() -> ()
  // TCL: set_location_assignment M20K_X15_Y9_N3 -to $parent|branch|leaf|foo|memBank2
}

// TCL-LABEL: proc regions_config
hw.module @regions() {
  %0 = hw.constant 0 : i1 { "loc:reg1" = #msft.switch.inst<@regions[]=#msft.logic_locked_region<region1, 0, 10, 0, 10>> }
  // TCL: set_instance_assignment -name PLACE_REGION "X0 Y0 X10 Y10" -to $parent|reg1
  // TCL: set_instance_assignment -name RESERVE_PLACE_REGION OFF -to $parent|reg1
  // TCL: set_instance_assignment -name CORE_ONLY_PLACE_REGION ON -to $parent|reg1
  // TCL: set_instance_assignment -name REGION_NAME region1 -to $parent|reg1

  %1 = hw.constant 1 : i1 { "loc:reg2" = #msft.switch.inst<@regions[]=#msft.logic_locked_region<region2, 0, 10, 10, 20>> }
  // TCL: set_instance_assignment -name PLACE_REGION "X0 Y10 X10 Y20" -to $parent|reg2
  // TCL: set_instance_assignment -name RESERVE_PLACE_REGION OFF -to $parent|reg2
  // TCL: set_instance_assignment -name CORE_ONLY_PLACE_REGION ON -to $parent|reg2
  // TCL: set_instance_assignment -name REGION_NAME region2 -to $parent|reg2  
}
