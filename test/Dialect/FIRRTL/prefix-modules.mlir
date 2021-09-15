// RUN: circt-opt --pass-pipeline="firrtl.circuit(firrtl-prefix-modules)" %s | FileCheck %s

// Check that the circuit is updated when the main module is updated.
// CHECK: firrtl.circuit "T_Top"
firrtl.circuit "Top" {
  // CHECK: firrtl.module @T_Top
  firrtl.module @Top()
    attributes {annotations = [{
      class = "sifive.enterprise.firrtl.NestedPrefixModulesAnnotation",
      prefix = "T_",
      inclusive = true
    }]} {
  }
}


// Check that the circuit is not updated if the annotation is non-inclusive.
// CHECK: firrtl.circuit "Top"
firrtl.circuit "Top" {
  // CHECK: firrtl.module @Top
  firrtl.module @Top()
    attributes {annotations = [{
      class = "sifive.enterprise.firrtl.NestedPrefixModulesAnnotation",
      prefix = "T_",
      inclusive = false
    }]} {
  }
}


// Check that basic module prefixing is working.
firrtl.circuit "Top" {
  // The annotation should be removed.
  // CHECK:  firrtl.module @Top() {
  firrtl.module @Top()
    attributes {annotations = [{
      class = "sifive.enterprise.firrtl.NestedPrefixModulesAnnotation",
      prefix = "T_",
      inclusive = false
    }]} {
    firrtl.instance @Zebra { name = "test" }
  }

  // CHECK: firrtl.module @T_Zebra
  // CHECK-NOT: firrtl.module @Zebra
  firrtl.module @Zebra() { }
}


// Check that memories are renamed.
firrtl.circuit "Top" {

  firrtl.module @Top()
    attributes {annotations = [{
      class = "sifive.enterprise.firrtl.NestedPrefixModulesAnnotation",
      prefix = "T_",
      inclusive = true
    }]} {
    // CHECK: name = "T_ram"
    %ram_ramport = firrtl.mem Undefined {depth = 256 : i64, name = "ram", portNames = ["ramport"], readLatency = 0 : i32, writeLatency = 1 : i32} : !firrtl.bundle<addr: uint<8>, en: uint<1>, clk: clock, data flip: uint<1>>
  }
}


// Check that external modules are not renamed.
// CHECK: firrtl.circuit "T_Top"
firrtl.circuit "Top" {
  // CHECK: firrtl.extmodule @ExternalModule
  firrtl.extmodule @ExternalModule()

  // CHECK: firrtl.module @T_Top
  firrtl.module @Top()
    attributes {annotations = [{
      class = "sifive.enterprise.firrtl.NestedPrefixModulesAnnotation",
      prefix = "T_",
      inclusive = true
    }]} {

    firrtl.instance @ExternalModule {name = "ext"}
  }
}


// Check that the module is not cloned more than necessary.
firrtl.circuit "Top0" {
  firrtl.module @Top0()
    attributes {annotations = [{
      class = "sifive.enterprise.firrtl.NestedPrefixModulesAnnotation",
      prefix = "T_",
      inclusive = false
    }]} {
    firrtl.instance @Zebra { name = "test" }
  }

  firrtl.module @Top1()
    attributes {annotations = [{
      class = "sifive.enterprise.firrtl.NestedPrefixModulesAnnotation",
      prefix = "T_",
      inclusive = false
    }]} {
    firrtl.instance @Zebra { name = "test" }
  }

  // CHECK: firrtl.module @T_Zebra
  // CHECK-NOT: firrtl.module @T_Zebra
  // CHECK-NOT: firrtl.module @Zebra
  firrtl.module @Zebra() { }
}


// Complex nested test.
// CHECK: firrtl.circuit "T_Top"
firrtl.circuit "Top" {
  // CHECK: firrtl.module @T_Top
  firrtl.module @Top()
    attributes {annotations = [{
      class = "sifive.enterprise.firrtl.NestedPrefixModulesAnnotation",
      prefix = "T_",
      inclusive = true
    }]} {

    // CHECK: firrtl.instance @T_Aardvark
    firrtl.instance @Aardvark { name = "test" }

    // CHECK: firrtl.instance @T_Z_Zebra
    firrtl.instance @Zebra { name = "test" }
  }

  // CHECK: firrtl.module @T_Aardvark
  firrtl.module @Aardvark()
    attributes {annotations = [{
      class = "sifive.enterprise.firrtl.NestedPrefixModulesAnnotation",
      prefix = "A_",
      inclusive = false
    }]} {

    // CHECK: firrtl.instance @T_A_Z_Zebra
    firrtl.instance @Zebra { name = "test" }
  }

  // CHECK: firrtl.module @T_Z_Zebra
  // CHECK: firrtl.module @T_A_Z_Zebra
  firrtl.module @Zebra() 
    attributes {annotations = [{
      class = "sifive.enterprise.firrtl.NestedPrefixModulesAnnotation",
      prefix = "Z_",
      inclusive = true
    }]} {
  }
}
