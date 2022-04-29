// RUN: circt-opt %s -verify-diagnostics | circt-opt -verify-diagnostics
// RUN: circt-opt %s --msft-lower-instances --verify-each | FileCheck %s
// RUN: circt-opt %s --msft-lower-instances --lower-msft-to-hw --lower-seq-to-sv --msft-export-tcl=tops=shallow,deeper,reg --export-verilog | FileCheck %s --check-prefix=TCL

msft.instance.hierarchy @deeper {
  msft.instance.dynamic @deeper::@branch {
    msft.instance.dynamic @shallow::@leaf {
      msft.instance.dynamic @leaf::@module {
        msft.pd.location M20K x: 15 y: 9 n: 3 path: "memBank2"
      }
    }
  }
}
// CHECK: hw.globalRef @instref [#hw.innerNameRef<@deeper::@branch>, #hw.innerNameRef<@shallow::@leaf>, #hw.innerNameRef<@leaf::@module>]
// CHECK: msft.pd.location @instref M20K x: 15 y: 9 n: 3 path : "memBank2"

msft.instance.hierarchy @shallow {
  msft.instance.dynamic @shallow::@leaf {
    msft.instance.dynamic @leaf::@module {
      msft.pd.location M20K x: 8 y: 19 n: 1 path: "memBank2"
    }
  }
}
// CHECK: hw.globalRef @instref_1 [#hw.innerNameRef<@shallow::@leaf>, #hw.innerNameRef<@leaf::@module>]
// CHECK: msft.pd.location @instref_1 M20K x: 8 y: 19 n: 1 path : "memBank2"

msft.instance.hierarchy @reg {
  msft.instance.dynamic @reg::@reg {
    msft.pd.location FF x: 0 y: 0 n: 0
  }
}
// CHECK: hw.globalRef @instref_2 [#hw.innerNameRef<@reg::@reg>]
// CHECK: msft.pd.location @instref_2 FF x: 0 y: 0 n: 0



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

// TCL-LABEL: proc reg_0_config
msft.module @reg {} (%input : i8, %clk : i1) -> () {
  %reg = seq.compreg sym @reg %input, %clk  : i8
  // TCL: set_location_assignment FF_X0_Y0_N0 -to $parent|reg_1
  msft.output
}
