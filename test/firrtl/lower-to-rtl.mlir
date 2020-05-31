// RUN: cirt-opt -lower-firrtl-to-rtl %s | FileCheck %s

// CHECK-LABEL: firrtl.module @Constant
firrtl.module @Constant(%in1: !firrtl.uint<4>,
                        %out1: !firrtl.flip<uint<4>>) {

  // CHECK: rtl.constant(-4 : i4) : i4
  %c12_ui4 = firrtl.constant(12 : ui4) : !firrtl.uint<4>
 
  // CHECK: rtl.constant(2 : i3) : i3
  %c2_si3 = firrtl.constant(2 : si3) : !firrtl.sint<3>

  // CHECK: %0 = rtl.add %c-4_i4, %c-4_i4 : i4
  %0 = firrtl.add %c12_ui4, %c12_ui4 : (!firrtl.uint<4>, !firrtl.uint<4>) -> !firrtl.uint<4>

  // CHECK-NEXT: firrtl.connect %out1, %0
  firrtl.connect %out1, %0 : !firrtl.flip<uint<4>>, !firrtl.uint<4>
}
