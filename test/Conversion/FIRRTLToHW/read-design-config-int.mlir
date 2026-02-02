// RUN: circt-opt --lower-firrtl-to-hw %s | FileCheck %s

firrtl.circuit "ReadDesignConfigIntTest" {
  // CHECK-LABEL: hw.module @ReadDesignConfigIntTest
  firrtl.module @ReadDesignConfigIntTest(
    out %param1: !firrtl.uint<4>,
    out %param2: !firrtl.uint<16>
  ) {
    // CHECK: %mode = sv.localparam {value = #hw.param.verbatim<"DesignConfigPackage::mode"> : i4} : i4
    %mode = firrtl.int.read_design_config_int "mode", 0, "Mode" : !firrtl.uint<4>

    // CHECK: %depth = sv.localparam {value = #hw.param.verbatim<"DesignConfigPackage::depth"> : i16} : i16
    %depth = firrtl.int.read_design_config_int "depth", 1024, "" : !firrtl.uint<16>

    // CHECK: hw.output %mode, %depth : i4, i16
    firrtl.connect %param1, %mode : !firrtl.uint<4>, !firrtl.uint<4>
    firrtl.connect %param2, %depth : !firrtl.uint<16>, !firrtl.uint<16>
  }

}

