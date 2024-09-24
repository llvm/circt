// RUN: circt-opt %s -test-firrtl-instance-info -split-input-file 2>&1 | FileCheck %s

// General test of DUT handling and layer handling when a `MarkDUTAnnotation` is
// present.
//
// CHECK:      "Foo"
// CHECK-NEXT:   hasDut: true
// CHECK-NEXT:   dut: firrtl.module private @Bar
// CHECK-NEXT:   effectiveDut: firrtl.module private @Bar
firrtl.circuit "Foo" {
  firrtl.layer @A bind {
  }
  // CHECK:      @Corge
  // CHECK-NEXT:   isDut: false
  // CHECK-NEXT:   atLeastOneInstanceUnderDut: false
  // CHECK-NEXT:   allInstancesUnderDut: false
  // CHECK-NEXT:   atLeastOneInstanceUnderLayer: true
  // CHECK-NEXT:   allInstancesUnderLayer: false
  firrtl.module private @Corge() {}
  // CHECK:      @Quz
  // CHECK-NEXT:   isDut: false
  // CHECK-NEXT:   atLeastOneInstanceUnderDut: false
  // CHECK-NEXT:   allInstancesUnderDut: false
  // CHECK-NEXT:   atLeastOneInstanceUnderLayer: true
  // CHECK-NEXT:   allInstancesUnderLayer: true
  firrtl.module private @Quz() {}
  // CHECK:      @Qux
  // CHECK-NEXT:   isDut: false
  // CHECK-NEXT:   atLeastOneInstanceUnderDut: true
  // CHECK-NEXT:   allInstancesUnderDut: false
  // CHECK-NEXT:   atLeastOneInstanceUnderLayer: false
  // CHECK-NEXT:   allInstancesUnderLayer: false
  firrtl.module private @Qux() {}
  // CHECK:      @Baz
  // CHECK-NEXT:   isDut: false
  // CHECK-NEXT:   atLeastOneInstanceUnderDut: true
  // CHECK-NEXT:   allInstancesUnderDut: true
  // CHECK-NEXT:   atLeastOneInstanceUnderLayer: false
  // CHECK-NEXT:   allInstancesUnderLayer: false
  firrtl.module private @Baz() {}
  // CHECK:      @Bar
  // CHECK-NEXT:   isDut: true
  // CHECK-NEXT:   atLeastOneInstanceUnderDut: true
  // CHECK-NEXT:   allInstancesUnderDut: true
  // CHECK-NEXT:   atLeastOneInstanceUnderLayer: false
  // CHECK-NEXT:   allInstancesUnderLayer: false
  firrtl.module private @Bar() attributes {
    annotations = [
      {class = "sifive.enterprise.firrtl.MarkDUTAnnotation"}
    ]
  } {
    firrtl.instance baz @Baz()
    firrtl.instance qux @Qux()
  }
  // CHECK:      @Foo
  // CHECK-NEXT:   isDut: false
  // CHECK-NEXT:   atLeastOneInstanceUnderDut: false
  // CHECK-NEXT:   allInstancesUnderDut: false
  // CHECK-NEXT:   atLeastOneInstanceUnderLayer: false
  // CHECK-NEXT:   allInstancesUnderLayer: false
  firrtl.module @Foo() {
    firrtl.instance bar @Bar()
    firrtl.instance qux @Qux()
    firrtl.layerblock @A {
      firrtl.instance quz @Quz()
      firrtl.instance corge @Corge()
    }
    firrtl.instance corge2 @Corge()
  }
}

// -----

// Test behavior when a `MarkDUTAnnotation` is absent.
//
// CHECK:      "Foo"
// CHECK-NEXT:   hasDut: false
// CHECK-NEXT:   dut: null
// CHECK-NEXT:   effectiveDut: firrtl.module @Foo
firrtl.circuit "Foo" {
  // CHECK:      @Foo
  // CHECK-NEXT:   isDut: false
  // CHECK-NEXT:   atLeastOneInstanceUnderDut: false
  // CHECK-NEXT:   allInstancesUnderDut: false
  // CHECK-NEXT:   atLeastOneInstanceUnderLayer: false
  // CHECK-NEXT:   allInstancesUnderLayer: false
  firrtl.module @Foo() {}
}

// -----

// Test that a non-FModuleOp can be queried.
firrtl.circuit "Foo" {
  // CHECK:      @Mem
  // CHECK-NEXT:   isDut: false
  // CHECK-NEXT:   atLeastOneInstanceUnderDut: true
  // CHECK-NEXT:   allInstancesUnderDut: true
  // CHECK-NEXT:   atLeastOneInstanceUnderLayer: false
  // CHECK-NEXT:   allInstancesUnderLayer: false
  firrtl.memmodule @Mem(
    in W0_addr: !firrtl.uint<1>,
    in W0_en: !firrtl.uint<1>,
    in W0_clk: !firrtl.clock,
    in W0_data: !firrtl.uint<8>
  ) attributes {
    dataWidth = 8 : ui32,
    depth = 2 : ui64,
    extraPorts = [],
    maskBits = 1 : ui32,
    numReadPorts = 0 : ui32,
    numReadWritePorts = 0 : ui32,
    numWritePorts = 1 : ui32,
    readLatency = 1 : ui32,
    writeLatency = 1 : ui32
  }
  firrtl.module @Foo() attributes {
    annotations = [
      {class = "sifive.enterprise.firrtl.MarkDUTAnnotation"}
    ]
  } {
    %0:4 = firrtl.instance Mem @Mem(
      in W0_addr: !firrtl.uint<1>,
      in W0_en: !firrtl.uint<1>,
      in W0_clk: !firrtl.clock,
      in W0_data: !firrtl.uint<8>
    )
  }
}
