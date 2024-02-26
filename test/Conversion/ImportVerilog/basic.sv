// RUN: circt-translate --import-verilog %s | FileCheck %s
// REQUIRES: slang

// Internal issue in Slang v3 about jump depending on uninitialised value.
// UNSUPPORTED: valgrind

// CHECK-LABEL: moore.module @Empty {
// CHECK:       }
module Empty;
  ; // empty member
endmodule

// CHECK-LABEL: moore.module @NestedA {
// CHECK:         moore.instance "NestedB" @NestedB
// CHECK:       }
// CHECK-LABEL: moore.module @NestedB {
// CHECK:         moore.instance "NestedC" @NestedC
// CHECK:       }
// CHECK-LABEL: moore.module @NestedC {
// CHECK:       }
module NestedA;
  module NestedB;
    module NestedC;
    endmodule
  endmodule
endmodule

// CHECK-LABEL: moore.module @Child {
// CHECK:       }
module Child;
endmodule

// CHECK-LABEL: moore.module @Parent
// CHECK:         moore.instance "child" @Child
// CHECK:       }
module Parent;
  Child child();
endmodule
