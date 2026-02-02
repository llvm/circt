// RUN: circt-opt --firrtl-create-design-config-package %s | FileCheck %s
firrtl.circuit "ReadDesignConfigIntTest" {
// CHECK:       sv.verbatim "package DesignConfigPackage
// CHECK-SAME: {output_file = #hw.output_file<"DesignConfigPackage.sv">}

// CHECK-LABEL:  firrtl.class @DesignConfigSchema(in %name_in: !firrtl.string, out %name: !firrtl.string, in %defaultValue_in: !firrtl.integer, out %defaultValue: !firrtl.integer, in %comment_in: !firrtl.string, out %comment: !firrtl.string, in %width_in: !firrtl.integer, out %width: !firrtl.integer) {
// CHECK-NEXT:    firrtl.propassign %name, %name_in : !firrtl.string
// CHECK-NEXT:    firrtl.propassign %defaultValue, %defaultValue_in : !firrtl.integer
// CHECK-NEXT:    firrtl.propassign %comment, %comment_in : !firrtl.string
// CHECK-NEXT:    firrtl.propassign %width, %width_in : !firrtl.integer
// CHECK-NEXT:  }

// CHECK-LABEL: firrtl.class @DesignConfigMetadata(out %configs: !firrtl.list<class<@DesignConfigSchema
// CHECK:          %mode = firrtl.object @DesignConfigSchema
// CHECK:          %depth = firrtl.object @DesignConfigSchema
// CHECK:          %[[list:.+]] = firrtl.list.create %mode, %depth
// CHECK:          firrtl.propassign %configs, %[[list]]
// CHECK:       }
  firrtl.module @ReadDesignConfigIntTest(
    out %param1: !firrtl.uint<4>,
    out %param2: !firrtl.uint<16>
  ) {
    %mode = firrtl.int.read_design_config_int "mode", 0, "Mode" : !firrtl.uint<4>

    %depth = firrtl.int.read_design_config_int "depth", 1024, "" : !firrtl.uint<16>

    firrtl.connect %param1, %mode : !firrtl.uint<4>, !firrtl.uint<4>
    firrtl.connect %param2, %depth : !firrtl.uint<16>, !firrtl.uint<16>
  }

}

