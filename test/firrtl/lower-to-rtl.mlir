// RUN: cirt-opt -lower-firrtl-to-rtl %s | FileCheck %s

// CHECK-LABEL: firrtl.module @Constant
firrtl.module @Constant() {

  // CHECK: rtl.constant(-4 : i4) : i4
  %c12_ui4 = firrtl.constant(12 : ui4) : !firrtl.uint<4>
 
  // CHECK: rtl.constant(2 : i3) : i3
  %c2_si3 = firrtl.constant(2 : si3) : !firrtl.sint<3>
}
