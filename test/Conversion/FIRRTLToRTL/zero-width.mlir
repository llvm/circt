// RUN: circt-opt -lower-firrtl-to-rtl %s | FileCheck %s

firrtl.circuit "Arithmetic" {
  // CHECK-LABEL: rtl.module @Arithmetic
  firrtl.module @Arithmetic(%uin3c: !firrtl.uint<3>,
                            %out0: !firrtl.flip<uint<3>>,
                            %out1: !firrtl.flip<uint<4>>,
                            %out2: !firrtl.flip<uint<4>>,
                            %out3: !firrtl.flip<uint<1>>) {
  %uin0c = firrtl.wire : !firrtl.uint<0>
  
    // CHECK-DAG: [[MULZERO:%.+]] = rtl.constant 0 : i3
    %0 = firrtl.mul %uin0c, %uin3c : (!firrtl.uint<0>, !firrtl.uint<3>) -> !firrtl.uint<3>
    firrtl.connect %out0, %0 : !firrtl.flip<uint<3>>, !firrtl.uint<3>

    // Lowers to nothing.
    %m0 = firrtl.mul %uin0c, %uin0c : (!firrtl.uint<0>, !firrtl.uint<0>) -> !firrtl.uint<0>

    // Lowers to nothing.
    %node = firrtl.node %m0 : !firrtl.uint<0>

    // Lowers to nothing.  Issue #429.
    %div = firrtl.div %node, %uin3c : (!firrtl.uint<0>, !firrtl.uint<3>) -> !firrtl.uint<0>

    // CHECK-DAG: %c0_i4 = rtl.constant 0 : i4
    // CHECK-DAG: %false = rtl.constant false
    // CHECK-NEXT: [[UIN3EXT:%.+]] = comb.concat %false, %uin3c : (i1, i3) -> i4
    // CHECK-NEXT: [[ADDRES:%.+]] = comb.add %c0_i4, [[UIN3EXT]] : i4
    %1 = firrtl.add %uin0c, %uin3c : (!firrtl.uint<0>, !firrtl.uint<3>) -> !firrtl.uint<4>
    firrtl.connect %out1, %1 : !firrtl.flip<uint<4>>, !firrtl.uint<4>

    %2 = firrtl.shl %node, 4 : (!firrtl.uint<0>) -> !firrtl.uint<4>
    firrtl.connect %out2, %2 : !firrtl.flip<uint<4>>, !firrtl.uint<4>

    // Issue #436
    %3 = firrtl.eq %uin0c, %uin0c : (!firrtl.uint<0>, !firrtl.uint<0>) -> !firrtl.uint<1>
    firrtl.connect %out3, %3 : !firrtl.flip<uint<1>>, !firrtl.uint<1>

    // CHECK: rtl.output %c0_i3, [[ADDRES]], %c0_i4, %true
  }

  // CHECK-LABEL: rtl.module @Exotic
  firrtl.module @Exotic(%uin3c: !firrtl.uint<3>,
                        %out0: !firrtl.flip<uint<3>>,
                        %out1: !firrtl.flip<uint<3>>) {
    %uin0c = firrtl.wire : !firrtl.uint<0>
  
    // CHECK-DAG: = rtl.constant true
    %0 = firrtl.andr %uin0c : (!firrtl.uint<0>) -> !firrtl.uint<1>

    // CHECK-DAG: = rtl.constant false
    %1 = firrtl.xorr %uin0c : (!firrtl.uint<0>) -> !firrtl.uint<1>

    %2 = firrtl.orr %uin0c : (!firrtl.uint<0>) -> !firrtl.uint<1>

    // Lowers to the uin3 value.
    %3 = firrtl.cat %uin0c, %uin3c : (!firrtl.uint<0>, !firrtl.uint<3>) -> !firrtl.uint<3>
    firrtl.connect %out0, %3 : !firrtl.flip<uint<3>>, !firrtl.uint<3>

    // Lowers to the uin3 value.
    %4 = firrtl.cat %uin3c, %uin0c : (!firrtl.uint<3>, !firrtl.uint<0>) -> !firrtl.uint<3>
    firrtl.connect %out1, %4 : !firrtl.flip<uint<3>>, !firrtl.uint<3>

    // Lowers to nothing.
    %5 = firrtl.cat %uin0c, %uin0c : (!firrtl.uint<0>, !firrtl.uint<0>) -> !firrtl.uint<0>

    // CHECK: rtl.output %uin3c, %uin3c : i3, i3
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

  // https://github.com/llvm/circt/issues/778
  firrtl.module @zero_width_mem(%clock: !firrtl.clock, %reset: !firrtl.uint<1>, %r0en: !firrtl.uint<1>) {
    %c0_ui4 = firrtl.constant(0 : i4) : !firrtl.uint<4>
    %c0_ui1 = firrtl.constant(false) : !firrtl.uint<1>
    %c0_ui25 = firrtl.constant(0 : i25) : !firrtl.uint<25>
    %tmp41_r0, %tmp41_w0 = firrtl.mem Undefined {depth = 10 : i64, name = "tmp41", portNames = ["r0", "w0"], readLatency = 0 : i32, writeLatency = 1 : i32} : !firrtl.bundle<addr: flip<uint<4>>, en: flip<uint<1>>, clk: flip<clock>, data: uint<0>>, !firrtl.flip<bundle<addr: uint<4>, en: uint<1>, clk: clock, data: uint<0>, mask: uint<1>>>
    %0 = firrtl.subfield %tmp41_r0("clk") : (!firrtl.bundle<addr: flip<uint<4>>, en: flip<uint<1>>, clk: flip<clock>, data: uint<0>>) -> !firrtl.flip<clock>
    firrtl.connect %0, %clock : !firrtl.flip<clock>, !firrtl.clock
    %1 = firrtl.subfield %tmp41_r0("en") : (!firrtl.bundle<addr: flip<uint<4>>, en: flip<uint<1>>, clk: flip<clock>, data: uint<0>>) -> !firrtl.flip<uint<1>>
    firrtl.connect %1, %r0en : !firrtl.flip<uint<1>>, !firrtl.uint<1>
    %2 = firrtl.subfield %tmp41_r0("addr") : (!firrtl.bundle<addr: flip<uint<4>>, en: flip<uint<1>>, clk: flip<clock>, data: uint<0>>) -> !firrtl.flip<uint<4>>
    firrtl.connect %2, %c0_ui4 : !firrtl.flip<uint<4>>, !firrtl.uint<4>
    %3 = firrtl.subfield %tmp41_w0("clk") : (!firrtl.flip<bundle<addr: uint<4>, en: uint<1>, clk: clock, data: uint<0>, mask: uint<1>>>) -> !firrtl.flip<clock>
    firrtl.connect %3, %clock : !firrtl.flip<clock>, !firrtl.clock
    %4 = firrtl.subfield %tmp41_w0("en") : (!firrtl.flip<bundle<addr: uint<4>, en: uint<1>, clk: clock, data: uint<0>, mask: uint<1>>>) -> !firrtl.flip<uint<1>>
    firrtl.connect %4, %r0en : !firrtl.flip<uint<1>>, !firrtl.uint<1>
    %5 = firrtl.subfield %tmp41_w0("addr") : (!firrtl.flip<bundle<addr: uint<4>, en: uint<1>, clk: clock, data: uint<0>, mask: uint<1>>>) -> !firrtl.flip<uint<4>>
    firrtl.connect %5, %c0_ui4 : !firrtl.flip<uint<4>>, !firrtl.uint<4>
    %6 = firrtl.subfield %tmp41_w0("mask") : (!firrtl.flip<bundle<addr: uint<4>, en: uint<1>, clk: clock, data: uint<0>, mask: uint<1>>>) -> !firrtl.flip<uint<1>>
    firrtl.connect %6, %c0_ui1 : !firrtl.flip<uint<1>>, !firrtl.uint<1>
    %7 = firrtl.subfield %tmp41_w0("data") : (!firrtl.flip<bundle<addr: uint<4>, en: uint<1>, clk: clock, data: uint<0>, mask: uint<1>>>) -> !firrtl.flip<uint<0>>
    firrtl.partialconnect %7, %c0_ui25 : !firrtl.flip<uint<0>>, !firrtl.uint<25>
  }
}
