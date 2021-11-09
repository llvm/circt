// RUN: circt-opt %s -verify-diagnostics | circt-opt -verify-diagnostics | FileCheck %s
// RUN: circt-opt %s --lower-msft-to-hw | FileCheck %s --check-prefix=LOWER
// RUN: circt-opt %s --lower-msft-to-hw --export-verilog | FileCheck %s --check-prefix=TCL

hw.module.extern @Foo()

// CHECK-LABEL: msft.module @leaf
// LOWER-LABEL: hw.module @leaf
msft.module @leaf {} () -> () {
  // CHECK: msft.instance @foo @Foo()
  // LOWER: hw.instance "foo" sym @foo @Foo() -> ()
  // LOWER-NOT: #msft.switch.inst
  msft.instance @foo @Foo() {
    "loc:memBank2" = #msft.switch.inst< @shallow["leaf"]=#msft.physloc<M20K, 8, 19, 1>,
                                        @deeper["branch","leaf"]=#msft.physloc<M20K, 15, 9, 3> >
  } : () -> ()
  // LOWER: sv.verbatim "proc shallow_config
  // LOWER: sv.verbatim "proc deeper_config
  msft.output
}

// TCL-NOT: proc leaf_config

// TCL-LABEL: proc shallow_config
msft.module @shallow {} () -> () {
  msft.instance @leaf @leaf() : () -> ()
  // TCL: set_location_assignment M20K_X8_Y19_N1 -to $parent|leaf|foo|memBank2
  msft.output
}

// TCL-LABEL: proc deeper_config
msft.module @deeper {} () -> () {
  msft.instance @branch @shallow() : () -> ()
  msft.instance @leaf @leaf() : () -> ()
  // TCL: set_location_assignment M20K_X15_Y9_N3 -to $parent|branch|leaf|foo|memBank2
  msft.output
}
