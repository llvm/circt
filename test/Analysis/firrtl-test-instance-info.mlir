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
  // CHECK-NEXT:   anyInstanceUnderDut: false
  // CHECK-NEXT:   allInstancesUnderDut: false
  // CHECK-NEXT:   anyInstanceUnderEffectiveDut: false
  // CHECK-NEXT:   allInstancesUnderEffectiveDut: false
  // CHECK-NEXT:   anyInstanceUnderLayer: true
  // CHECK-NEXT:   allInstancesUnderLayer: false
  firrtl.module private @Corge() {}
  // CHECK:      @Quz
  // CHECK-NEXT:   isDut: false
  // CHECK-NEXT:   anyInstanceUnderDut: false
  // CHECK-NEXT:   allInstancesUnderDut: false
  // CHECK-NEXT:   anyInstanceUnderEffectiveDut: false
  // CHECK-NEXT:   allInstancesUnderEffectiveDut: false
  // CHECK-NEXT:   anyInstanceUnderLayer: true
  // CHECK-NEXT:   allInstancesUnderLayer: true
  firrtl.module private @Quz() {}
  // CHECK:      @Qux
  // CHECK-NEXT:   isDut: false
  // CHECK-NEXT:   anyInstanceUnderDut: true
  // CHECK-NEXT:   allInstancesUnderDut: false
  // CHECK-NEXT:   anyInstanceUnderEffectiveDut: true
  // CHECK-NEXT:   allInstancesUnderEffectiveDut: false
  // CHECK-NEXT:   anyInstanceUnderLayer: false
  // CHECK-NEXT:   allInstancesUnderLayer: false
  firrtl.module private @Qux() {}
  // CHECK:      @Baz
  // CHECK-NEXT:   isDut: false
  // CHECK-NEXT:   anyInstanceUnderDut: true
  // CHECK-NEXT:   allInstancesUnderDut: true
  // CHECK-NEXT:   anyInstanceUnderEffectiveDut: true
  // CHECK-NEXT:   allInstancesUnderEffectiveDut: true
  // CHECK-NEXT:   anyInstanceUnderLayer: false
  // CHECK-NEXT:   allInstancesUnderLayer: false
  firrtl.module private @Baz() {}
  // CHECK:      @Bar
  // CHECK-NEXT:   isDut: true
  // CHECK-NEXT:   anyInstanceUnderDut: true
  // CHECK-NEXT:   allInstancesUnderDut: true
  // CHECK-NEXT:   anyInstanceUnderEffectiveDut: true
  // CHECK-NEXT:   allInstancesUnderEffectiveDut: true
  // CHECK-NEXT:   anyInstanceUnderLayer: false
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
  // CHECK-NEXT:   anyInstanceUnderDut: false
  // CHECK-NEXT:   allInstancesUnderDut: false
  // CHECK-NEXT:   anyInstanceUnderEffectiveDut: false
  // CHECK-NEXT:   allInstancesUnderEffectiveDut: false
  // CHECK-NEXT:   anyInstanceUnderLayer: false
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
  // CHECK-NEXT:   anyInstanceUnderDut: false
  // CHECK-NEXT:   allInstancesUnderDut: false
  // CHECK-NEXT:   anyInstanceUnderEffectiveDut: true
  // CHECK-NEXT:   allInstancesUnderEffectiveDut: true
  // CHECK-NEXT:   anyInstanceUnderLayer: false
  // CHECK-NEXT:   allInstancesUnderLayer: false
  firrtl.module @Foo() {}
}

// -----

// Test that a non-FModuleOp can be queried.
firrtl.circuit "Foo" {
  // CHECK:      @Mem
  // CHECK-NEXT:   isDut: false
  // CHECK-NEXT:   anyInstanceUnderDut: true
  // CHECK-NEXT:   allInstancesUnderDut: true
  // CHECK-NEXT:   anyInstanceUnderEffectiveDut: true
  // CHECK-NEXT:   allInstancesUnderEffectiveDut: true
  // CHECK-NEXT:   anyInstanceUnderLayer: false
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

// -----

// Test that if the top module is the DUT that it still gets marked as "under"
// the DUT.
firrtl.circuit "Foo" {
  // CHECK:      @Foo
  // CHECK-NEXT:   isDut: true
  // CHECK-NEXT:   anyInstanceUnderDut: true
  // CHECK-NEXT:   allInstancesUnderDut: true
  // CHECK-NEXT:   anyInstanceUnderEffectiveDut: true
  // CHECK-NEXT:   allInstancesUnderEffectiveDut: true
  firrtl.module @Foo() attributes {
    annotations = [
      {class = "sifive.enterprise.firrtl.MarkDUTAnnotation"}
    ]
  } {}
}

// -----

// Test that "inDesign" works properly.  This test instantiates Baz twice.  Once
// under the testbench, Foo, and once under a layer in the design-under-test
// (DUT), Bar.  Baz is under the DUT, but not in the design.
firrtl.circuit "Foo" {
  firrtl.layer @A bind {}
  // CHECK:      @Baz
  // CHECK-NEXT:   isDut: false
  // CHECK-NEXT:   anyInstanceUnderDut: true
  // CHECK-NEXT:   allInstancesUnderDut: false
  // CHECK-NEXT:   anyInstanceUnderEffectiveDut: true
  // CHECK-NEXT:   allInstancesUnderEffectiveDut: false
  // CHECK-NEXT:   anyInstanceUnderLayer: true
  // CHECK-NEXT:   allInstancesUnderLayer: false
  // CHECK-NEXT:   anyInstanceInDesign: false
  // CHECK-NEXT:   allInstancesInDesign: false
  // CHECK-NEXT:   anyInstanceInEffectiveDesign: false
  // CHECK-NEXT:   allInstancesInEffectiveDesign: false
  firrtl.module private @Baz() {}
  // CHECK:      @Bar
  // CHECK-NEXT:   isDut: true
  // CHECK-NEXT:   anyInstanceUnderDut: true
  // CHECK-NEXT:   allInstancesUnderDut: true
  // CHECK-NEXT:   anyInstanceUnderEffectiveDut: true
  // CHECK-NEXT:   allInstancesUnderEffectiveDut: true
  // CHECK-NEXT:   anyInstanceUnderLayer: false
  // CHECK-NEXT:   allInstancesUnderLayer: false
  // CHECK-NEXT:   anyInstanceInDesign: true
  // CHECK-NEXT:   allInstancesInDesign: true
  // CHECK-NEXT:   anyInstanceInEffectiveDesign: true
  // CHECK-NEXT:   allInstancesInEffectiveDesign: true
  firrtl.module @Bar() attributes {
    annotations = [
      {
        class = "sifive.enterprise.firrtl.MarkDUTAnnotation"
      }
    ]
  } {
    firrtl.layerblock @A {
      firrtl.instance baz @Baz()
    }
  }
  // CHECK:      @Foo
  // CHECK-NEXT:   isDut: false
  // CHECK-NEXT:   anyInstanceUnderDut: false
  // CHECK-NEXT:   allInstancesUnderDut: false
  // CHECK-NEXT:   anyInstanceUnderEffectiveDut: false
  // CHECK-NEXT:   allInstancesUnderEffectiveDut: false
  // CHECK-NEXT:   anyInstanceUnderLayer: false
  // CHECK-NEXT:   allInstancesUnderLayer: false
  // CHECK-NEXT:   anyInstanceInDesign: false
  // CHECK-NEXT:   allInstancesInDesign: false
  // CHECK-NEXT:   anyInstanceInEffectiveDesign: false
  // CHECK-NEXT:   allInstancesInEffectiveDesign: false
  firrtl.module @Foo() {
    firrtl.instance dut interesting_name @Bar()
    firrtl.instance foo interesting_name @Baz()
  }
}

// -----

// Test that "inDesign" works properly for the effective DUT.
firrtl.circuit "Foo" {
  firrtl.layer @A bind {}
  // CHECK:      @Baz
  // CHECK:        anyInstanceInDesign: false
  // CHECK-NEXT:   allInstancesInDesign: false
  // CHECK-NEXT:   anyInstanceInEffectiveDesign: true
  // CHECK-NEXT:   allInstancesInEffectiveDesign: false
  firrtl.module private @Baz() {}
  // CHECK:      @Bar
  // CHECK:        anyInstanceInDesign: false
  // CHECK-NEXT:   allInstancesInDesign: false
  // CHECK-NEXT:   anyInstanceInEffectiveDesign: true
  // CHECK-NEXT:   allInstancesInEffectiveDesign: true
  firrtl.module @Bar() {
    firrtl.layerblock @A {
      firrtl.instance baz @Baz()
    }
  }
  // CHECK:      @Foo
  // CHECK:        anyInstanceInDesign: false
  // CHECK-NEXT:   allInstancesInDesign: false
  // CHECK-NEXT:   anyInstanceInEffectiveDesign: true
  // CHECK-NEXT:   allInstancesInEffectiveDesign: true
  firrtl.module @Foo() {
    firrtl.instance dut interesting_name @Bar()
    firrtl.instance foo interesting_name @Baz()
  }
}

// -----

// Test that modules that are not instantiated are put in the effective design.
firrtl.circuit "Foo" {
  // CHECK:      @Bar
  // CHECK:        anyInstanceInEffectiveDesign: true
  // CHECK-NEXT:   allInstancesInEffectiveDesign: true
  firrtl.module @Bar() {}
  // CHECK: @Foo
  firrtl.module @Foo() {}
}

// -----

// A circuit with a testharness, a DUT, and a Grand Central companion.  Test
// that all instance combinations are correct.
firrtl.circuit "TestharnessHasDUT" {

  // Each of these modules is instantiated in a different location.  The
  // instantiation location is indicated by three binary bits with an "_"
  // indicating the absence of instantiation:
  //   1) "T" indicates this is instantiated in the "Testharness"
  //   2) "D" indicates this is instantiated in the "DUT"
  //   3) "C" indicates this is instantiated in the "Companion"
  // E.g., "T_C" is an module instantiated above the DUT and in the Companion.

  // CHECK:      @T__
  // CHECK:        anyInstanceInDesign: false
  // CHECK-NEXT:   allInstancesInDesign: false
  // CHECK-NEXT:   anyInstanceInEffectiveDesign: false
  // CHECK-NEXT:   allInstancesInEffectiveDesign: false
  firrtl.module @T__() {}
  // CHECK:      @_D_
  // CHECK:        anyInstanceInDesign: true
  // CHECK-NEXT:   allInstancesInDesign: true
  // CHECK-NEXT:   anyInstanceInEffectiveDesign: true
  // CHECK-NEXT:   allInstancesInEffectiveDesign: true
  firrtl.module @_D_() {}
  // CHECK:      @__C
  // CHECK:        anyInstanceInDesign: false
  // CHECK-NEXT:   allInstancesInDesign: false
  // CHECK-NEXT:   anyInstanceInEffectiveDesign: false
  // CHECK-NEXT:   allInstancesInEffectiveDesign: false
  firrtl.module @__C() {}
  // CHECK:      @TD_
  // CHECK:        anyInstanceInDesign: true
  // CHECK-NEXT:   allInstancesInDesign: false
  // CHECK-NEXT:   anyInstanceInEffectiveDesign: true
  // CHECK-NEXT:   allInstancesInEffectiveDesign: false
  firrtl.module @TD_() {}
  // CHECK:      @_DC
  // CHECK:        anyInstanceInDesign: true
  // CHECK-NEXT:   allInstancesInDesign: false
  // CHECK-NEXT:   anyInstanceInEffectiveDesign: true
  // CHECK-NEXT:   allInstancesInEffectiveDesign: false
  firrtl.module @_DC() {}
  // CHECK:      @T_C
  // CHECK:        anyInstanceInDesign: false
  // CHECK-NEXT:   allInstancesInDesign: false
  // CHECK-NEXT:   anyInstanceInEffectiveDesign: false
  // CHECK-NEXT:   allInstancesInEffectiveDesign: false
  firrtl.module @T_C() {}
  // CHECK:      @TDC
  // CHECK:        anyInstanceInDesign: true
  // CHECK-NEXT:   allInstancesInDesign: false
  // CHECK-NEXT:   anyInstanceInEffectiveDesign: true
  // CHECK-NEXT:   allInstancesInEffectiveDesign: false
  firrtl.module @TDC() {}

  firrtl.module private @Companion() attributes {
    annotations = [
      {
        class = "sifive.enterprise.grandcentral.ViewAnnotation.companion",
        defName = "Foo",
        id = 0 : i64,
        name = "View"
      }
    ]
  } {

    firrtl.instance m__c @__C()
    firrtl.instance m_dc @_DC()
    firrtl.instance mt_c @T_C()
    firrtl.instance mtdc @TDC()
  }

  firrtl.module private @DUT() attributes {
    annotations = [
      {
        class = "sifive.enterprise.firrtl.MarkDUTAnnotation"
      }
    ]
  } {
    firrtl.instance companion @Companion()

    firrtl.instance m_d_ @_D_()
    firrtl.instance mtd_ @TD_()
    firrtl.instance m_dc @_DC()
    firrtl.instance mtdc @TDC()
  }

  // The Top module that instantiates the DUT
  firrtl.module @TestharnessHasDUT() {
    firrtl.instance dut @DUT()

    firrtl.instance mt__ @T__()
    firrtl.instance mtd_ @TD_()
    firrtl.instance mt_c @T_C()
    firrtl.instance mtdc @TDC()
  }
}

// -----

// This is the same as the previous circuit excpet there is no DUT specified.
// This tests that the differences between "design" and "effective design" are
// correctly captured.
firrtl.circuit "TestharnessNoDUT" {

  // Each of these modules is instantiated in a different location.  The
  // instantiation location is indicated by three binary bits with an "_"
  // indicating the absence of instantiation:
  //   1) "T" indicates this is instantiated in the "Testharness"
  //   2) "D" indicates this is instantiated in the "DUT"
  //   3) "C" indicates this is instantiated in the "Companion"
  // E.g., "T_C" is an module instantiated above the DUT and in the Companion.

  // CHECK:      @T__
  // CHECK:        anyInstanceInDesign: false
  // CHECK-NEXT:   allInstancesInDesign: false
  // CHECK-NEXT:   anyInstanceInEffectiveDesign: true
  // CHECK-NEXT:   allInstancesInEffectiveDesign: true
  firrtl.module @T__() {}
  // CHECK:      @_D_
  // CHECK:        anyInstanceInDesign: false
  // CHECK-NEXT:   allInstancesInDesign: false
  // CHECK-NEXT:   anyInstanceInEffectiveDesign: true
  // CHECK-NEXT:   allInstancesInEffectiveDesign: true
  firrtl.module @_D_() {}
  // CHECK:      @__C
  // CHECK:        anyInstanceInDesign: false
  // CHECK-NEXT:   allInstancesInDesign: false
  // CHECK-NEXT:   anyInstanceInEffectiveDesign: false
  // CHECK-NEXT:   allInstancesInEffectiveDesign: false
  firrtl.module @__C() {}
  // CHECK:      @TD_
  // CHECK:        anyInstanceInDesign: false
  // CHECK-NEXT:   allInstancesInDesign: false
  // CHECK-NEXT:   anyInstanceInEffectiveDesign: true
  // CHECK-NEXT:   allInstancesInEffectiveDesign: true
  firrtl.module @TD_() {}
  // CHECK:      @_DC
  // CHECK:        anyInstanceInDesign: false
  // CHECK-NEXT:   allInstancesInDesign: false
  // CHECK-NEXT:   anyInstanceInEffectiveDesign: true
  // CHECK-NEXT:   allInstancesInEffectiveDesign: false
  firrtl.module @_DC() {}
  // CHECK:      @T_C
  // CHECK:        anyInstanceInDesign: false
  // CHECK-NEXT:   allInstancesInDesign: false
  // CHECK-NEXT:   anyInstanceInEffectiveDesign: true
  // CHECK-NEXT:   allInstancesInEffectiveDesign: false
  firrtl.module @T_C() {}
  // CHECK:      @TDC
  // CHECK:        anyInstanceInDesign: false
  // CHECK-NEXT:   allInstancesInDesign: false
  // CHECK-NEXT:   anyInstanceInEffectiveDesign: true
  // CHECK-NEXT:   allInstancesInEffectiveDesign: false
  firrtl.module @TDC() {}

  firrtl.module private @Companion() attributes {
    annotations = [
      {
        class = "sifive.enterprise.grandcentral.ViewAnnotation.companion",
        defName = "Foo",
        id = 0 : i64,
        name = "View"
      }
    ]
  } {

    firrtl.instance m__c @__C()
    firrtl.instance m_dc @_DC()
    firrtl.instance mt_c @T_C()
    firrtl.instance mtdc @TDC()
  }

  firrtl.module private @DUT() {
    firrtl.instance companion @Companion()

    firrtl.instance m_d_ @_D_()
    firrtl.instance mtd_ @TD_()
    firrtl.instance m_dc @_DC()
    firrtl.instance mtdc @TDC()
  }

  // The Top module that instantiates the DUT
  firrtl.module @TestharnessNoDUT() {
    firrtl.instance dut @DUT()

    firrtl.instance mt__ @T__()
    firrtl.instance mtd_ @TD_()
    firrtl.instance mt_c @T_C()
    firrtl.instance mtdc @TDC()
  }
}

// -----

// Test that `lowerToBind` is treated the same as a layer.  This is important
// because it ensures that the InstanceInfo analysis behaves the same before or
// after the `LowerLayers` pass is run.
firrtl.circuit "Foo" {
  // CHECK:      @Bar
  // CHECK:        anyInstanceUnderLayer: true
  // CHECK-NEXT:   allInstancesUnderLayer: true
  // CHECK:        anyInstanceInEffectiveDesign: false
  // CHECK-NEXT:   allInstancesInEffectiveDesign: false
  firrtl.module @Bar() {
  }
  // CHECK:      @Foo_A
  // CHECK:        anyInstanceUnderLayer: true
  // CHECK-NEXT:   allInstancesUnderLayer: true
  // CHECK:        anyInstanceInEffectiveDesign: false
  // CHECK-NEXT:   allInstancesInEffectiveDesign: false
  firrtl.module @Foo_A() {
    firrtl.instance bar @Bar()
  }
  // CHECK:      @Foo
  firrtl.module @Foo() {
    firrtl.instance a {lowerToBind} @Foo_A()
  }
}

// -----

// Test that the DUT can be an extmodule
// CHECK:      - operation: firrtl.circuit "Testharness"
// CHECK-NEXT:   hasDut: true
// CHECK-NEXT:   dut: firrtl.extmodule private @DUT
// CHECK-NEXT:   effectiveDut: firrtl.extmodule private @DUT
firrtl.circuit "Testharness" {
  // CHECK:      - operation: firrtl.module @Testharness
  // CHECK-NEXT:   isDut: false
  // CHECK-NEXT:   anyInstanceUnderDut: false
  // CHECK-NEXT:   allInstancesUnderDut: false
  firrtl.module @Testharness() {
    firrtl.instance dut @DUT()
    firrtl.instance foo @Foo()
  }

  // CHECK:      - operation: firrtl.extmodule private @DUT
  // CHECK-NEXT:   isDut: true
  // CHECK-NEXT:   anyInstanceUnderDut: true
  // CHECK-NEXT:   allInstancesUnderDut: true
  firrtl.extmodule private @DUT() attributes {
    annotations = [
      {
        class = "sifive.enterprise.firrtl.MarkDUTAnnotation"
      }
    ]
  }

  // CHECK:      - operation: firrtl.module private @Foo
  // CHECK-NEXT:   isDut: false
  // CHECK-NEXT:   anyInstanceUnderDut: false
  // CHECK-NEXT:   allInstancesUnderDut: false
  firrtl.module private @Foo() {
  }
}
