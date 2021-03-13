// RUN: circt-opt -lower-firrtl-to-rtl %s | FileCheck %s

firrtl.circuit "Arithmetic" {
  // CHECK-LABEL: rtl.module @Arithmetic
  firrtl.module @Arithmetic(%uin3c: !firrtl.uint<3>,
                            %out0: !firrtl.flip<uint<3>>,
                            %out1: !firrtl.flip<uint<4>>,
                            %out2: !firrtl.flip<uint<4>>,
                            %out3: !firrtl.flip<uint<1>>) {
    %uin0c = firrtl.wire : !firrtl.uint<0>
  
    // CHECK-NEXT: rtl.constant 0 : i3
    // CHECK-NEXT: [[MULZERO:%.+]] = rtl.constant 0 : i3
    %0 = firrtl.mul %uin0c, %uin3c : (!firrtl.uint<0>, !firrtl.uint<3>) -> !firrtl.uint<3>
    firrtl.connect %out0, %0 : !firrtl.flip<uint<3>>, !firrtl.uint<3>

    // Lowers to nothing.
    %m0 = firrtl.mul %uin0c, %uin0c : (!firrtl.uint<0>, !firrtl.uint<0>) -> !firrtl.uint<0>

    // Lowers to nothing.
    %node = firrtl.node %m0 : !firrtl.uint<0>

    // Lowers to nothing.  Issue #429.
    %div = firrtl.div %node, %uin3c : (!firrtl.uint<0>, !firrtl.uint<3>) -> !firrtl.uint<0>

    // CHECK-NEXT: %c0_i4 = rtl.constant 0 : i4
    // CHECK-NEXT: %false = rtl.constant false
    // CHECK-NEXT: [[UIN3EXT:%.+]] = comb.concat %false, %uin3c : (i1, i3) -> i4
    // CHECK-NEXT: [[ADDRES:%.+]] = comb.add %c0_i4, [[UIN3EXT]] : i4
    %1 = firrtl.add %uin0c, %uin3c : (!firrtl.uint<0>, !firrtl.uint<3>) -> !firrtl.uint<4>
    firrtl.connect %out1, %1 : !firrtl.flip<uint<4>>, !firrtl.uint<4>

    // CHECK-NEXT: [[SHL:%.+]] = rtl.constant 0 : i4
    %2 = firrtl.shl %node, 4 : (!firrtl.uint<0>) -> !firrtl.uint<4>
    firrtl.connect %out2, %2 : !firrtl.flip<uint<4>>, !firrtl.uint<4>

    // Issue #436
    // CHECK: [[CMP:%true.*]] = rtl.constant true
    %3 = firrtl.eq %uin0c, %uin0c : (!firrtl.uint<0>, !firrtl.uint<0>) -> !firrtl.uint<1>
    firrtl.connect %out3, %3 : !firrtl.flip<uint<1>>, !firrtl.uint<1>
  }

  // CHECK-LABEL: rtl.module @Exotic
  firrtl.module @Exotic(%uin3c: !firrtl.uint<3>,
                        %out0: !firrtl.flip<uint<3>>,
                        %out1: !firrtl.flip<uint<3>>) {
    %uin0c = firrtl.wire : !firrtl.uint<0>
  
    // CHECK-NEXT: = rtl.constant true
    %0 = firrtl.andr %uin0c : (!firrtl.uint<0>) -> !firrtl.uint<1>

    // CHECK-NEXT: = rtl.constant false
    %1 = firrtl.xorr %uin0c : (!firrtl.uint<0>) -> !firrtl.uint<1>

    // CHECK-NEXT: = rtl.constant false
    %2 = firrtl.orr %uin0c : (!firrtl.uint<0>) -> !firrtl.uint<1>

    // Lowers to the uin3 value.
    %3 = firrtl.cat %uin0c, %uin3c : (!firrtl.uint<0>, !firrtl.uint<3>) -> !firrtl.uint<3>
    firrtl.connect %out0, %3 : !firrtl.flip<uint<3>>, !firrtl.uint<3>

    // Lowers to the uin3 value.
    %4 = firrtl.cat %uin3c, %uin0c : (!firrtl.uint<3>, !firrtl.uint<0>) -> !firrtl.uint<3>
    firrtl.connect %out1, %4 : !firrtl.flip<uint<3>>, !firrtl.uint<3>

    // Lowers to nothing.
    %5 = firrtl.cat %uin0c, %uin0c : (!firrtl.uint<0>, !firrtl.uint<0>) -> !firrtl.uint<0>
  }

  // CHECK-LABEL: rtl.module @Decls
  firrtl.module @Decls(%uin3c: !firrtl.uint<3>) {
    %sin0c = firrtl.wire : !firrtl.sint<0>
    %uin0c = firrtl.wire : !firrtl.uint<0>

    // Lowers to nothing.
    %wire = firrtl.wire : !firrtl.flip<sint<0>>
    firrtl.connect %wire, %sin0c : !firrtl.flip<sint<0>>, !firrtl.sint<0>

    // CHECK-NEXT: rtl.output
  }
}
