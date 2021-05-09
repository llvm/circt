// RUN: circt-opt -pass-pipeline='firrtl.circuit(firrtl-imconstprop)' %s | FileCheck %s

firrtl.circuit "Test" {

  // CHECK-LABEL: @PassThrough
  // CHECK: (%source: !firrtl.uint<1>, %dest: !firrtl.flip<uint<1>>) 
  firrtl.module @PassThrough(%source: !firrtl.uint<1>, %dest: !firrtl.flip<uint<1>>) {
    // CHECK-NEXT: %c0_ui1 = firrtl.constant(0 : ui1) : !firrtl.uint<1>
    // CHECK-NEXT: firrtl.connect %dest, %c0_ui1
    firrtl.connect %dest, %source : !firrtl.flip<uint<1>>, !firrtl.uint<1>
    // CHECK-NEXT: }
  }

  // CHECK-LABEL: @Test
  firrtl.module @Test(%result1: !firrtl.flip<uint<1>>,
                      %result2: !firrtl.flip<uint<1>>,
                      %result3: !firrtl.flip<uint<1>>,
                      %result4: !firrtl.flip<uint<2>>) {
    %c0_ui1 = firrtl.constant(0 : ui1) : !firrtl.uint<1>
    %c1_ui1 = firrtl.constant(1 : ui1) : !firrtl.uint<1>

    // Trivial wire constant propagation.
    %someWire = firrtl.wire : !firrtl.uint<1>
    firrtl.connect %someWire, %c0_ui1 : !firrtl.uint<1>, !firrtl.uint<1>

    // CHECK-NOT: firrtl.wire
    // CHECK: firrtl.connect %result1, %c0_ui1_0
    firrtl.connect %result1, %someWire : !firrtl.flip<uint<1>>, !firrtl.uint<1>

    // Not a constant.
    %nonconstWire = firrtl.wire : !firrtl.uint<1>
    firrtl.connect %nonconstWire, %c0_ui1 : !firrtl.uint<1>, !firrtl.uint<1>
    firrtl.connect %nonconstWire, %c1_ui1 : !firrtl.uint<1>, !firrtl.uint<1>

    // CHECK: firrtl.connect %result2, %nonconstWire
    firrtl.connect %result2, %nonconstWire : !firrtl.flip<uint<1>>, !firrtl.uint<1>


    // Constant propagation through instance.
    %source, %dest = firrtl.instance @PassThrough {name = "", portNames = ["source", "dest"]} : !firrtl.flip<uint<1>>, !firrtl.uint<1>

    // CHECK: firrtl.connect %inst_source, %c0_ui1
    firrtl.connect %source, %c0_ui1 : !firrtl.flip<uint<1>>, !firrtl.uint<1>
    // CHECK: firrtl.connect %result3, %c0_ui1_1
    firrtl.connect %result3, %dest : !firrtl.flip<uint<1>>, !firrtl.uint<1>

    // Check connect extensions.
    %extWire = firrtl.wire : !firrtl.uint<2>
    firrtl.connect %extWire, %c0_ui1 : !firrtl.uint<2>, !firrtl.uint<1>

    // CHECK: firrtl.connect %result4, %c0_ui2
    firrtl.connect %result4, %extWire: !firrtl.flip<uint<2>>, !firrtl.uint<2>
  }

  // Unused modules should be completely dropped.

  // CHECK-LABEL: @UnusedModule(%source: !firrtl.uint<1>, %dest: !firrtl.flip<uint<1>>)
  firrtl.module @UnusedModule(%source: !firrtl.uint<1>, %dest: !firrtl.flip<uint<1>>) {
    firrtl.connect %dest, %source : !firrtl.flip<uint<1>>, !firrtl.uint<1>
    // CHECK-NEXT: }
  }
}

