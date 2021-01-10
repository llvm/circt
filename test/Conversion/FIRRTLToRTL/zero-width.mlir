// RUN: circt-opt -lower-firrtl-to-rtl %s | FileCheck %s

module attributes {firrtl.mainModule = "Simple"} {
   // CHECK-LABEL: rtl.module @Arithmetic
  rtl.module @Arithmetic(%sin0: i0, %uin0: i0, %uin3: i3) -> (i3, i4, i4) {
    %sin0c = firrtl.stdIntCast %sin0 : (i0) -> !firrtl.sint<0>
    %uin0c = firrtl.stdIntCast %uin0 : (i0) -> !firrtl.uint<0>
    %uin3c = firrtl.stdIntCast %uin3 : (i3) -> !firrtl.uint<3>
  
    // CHECK-NEXT: rtl.constant(0 : i3) : i3
    // CHECK-NEXT: [[MULZERO:%.+]] = rtl.constant(0 : i3) : i3
    %0 = firrtl.mul %uin0c, %uin3c : (!firrtl.uint<0>, !firrtl.uint<3>) -> !firrtl.uint<3>
    %c0 = firrtl.stdIntCast %0 : (!firrtl.uint<3>) -> i3

    // Lowers to nothing.
    %m0 = firrtl.mul %uin0c, %uin0c : (!firrtl.uint<0>, !firrtl.uint<0>) -> !firrtl.uint<0>

    // Lowers to nothing.
    %node = firrtl.node %m0 : !firrtl.uint<0>

    // CHECK-NEXT: %c0_i4 = rtl.constant(0 : i4) : i4
    // CHECK-NEXT: [[UIN3EXT:%.+]] = rtl.zext %uin3 : (i3) -> i4
    // CHECK-NEXT: [[ADDRES:%.+]] = rtl.add %c0_i4, [[UIN3EXT]] : i4
    %1 = firrtl.add %uin0c, %uin3c : (!firrtl.uint<0>, !firrtl.uint<3>) -> !firrtl.uint<4>
    %c1 = firrtl.stdIntCast %1 : (!firrtl.uint<4>) -> i4

    // CHECK-NEXT: [[SHL:%.+]] = rtl.constant(0 : i4) : i4
    %2 = firrtl.shl %node, 4 : (!firrtl.uint<0>) -> !firrtl.uint<4>
    %c2 = firrtl.stdIntCast %2 : (!firrtl.uint<4>) -> i4

    // CHECK-NEXT: rtl.output [[MULZERO]], [[ADDRES]], [[SHL]] : i3, i4, i4
    rtl.output %c0, %c1, %c2 : i3, i4, i4
  }

  // CHECK-LABEL: rtl.module @Exotic
  rtl.module @Exotic(%sin0: i0, %uin0: i0, %uin3: i3) -> (i3, i3) {
    %sin0c = firrtl.stdIntCast %sin0 : (i0) -> !firrtl.sint<0>
    %uin0c = firrtl.stdIntCast %uin0 : (i0) -> !firrtl.uint<0>
    %uin3c = firrtl.stdIntCast %uin3 : (i3) -> !firrtl.uint<3>
  
    // CHECK-NEXT: = rtl.constant(true) 
    %0 = firrtl.andr %uin0c : (!firrtl.uint<0>) -> !firrtl.uint<1>

    // CHECK-NEXT: = rtl.constant(false) 
    %1 = firrtl.xorr %uin0c : (!firrtl.uint<0>) -> !firrtl.uint<1>

    // CHECK-NEXT: = rtl.constant(false)
    %2 = firrtl.orr %uin0c : (!firrtl.uint<0>) -> !firrtl.uint<1>

    // Lowers to the uin3 value.
    %3 = firrtl.cat %uin0c, %uin3c : (!firrtl.uint<0>, !firrtl.uint<3>) -> !firrtl.uint<3>
    %c3 = firrtl.stdIntCast %3 : (!firrtl.uint<3>) -> i3

    // Lowers to the uin3 value.
    %4 = firrtl.cat %uin3c, %uin0c : (!firrtl.uint<3>, !firrtl.uint<0>) -> !firrtl.uint<3>
    %c4 = firrtl.stdIntCast %4 : (!firrtl.uint<3>) -> i3

    // Lowers to nothing.
    %5 = firrtl.cat %uin0c, %uin0c : (!firrtl.uint<0>, !firrtl.uint<0>) -> !firrtl.uint<0>

    // CHECK-NEXT: rtl.output %uin3, %uin3 : i3, i3
    rtl.output %c3, %c4 : i3, i3
  }
}
