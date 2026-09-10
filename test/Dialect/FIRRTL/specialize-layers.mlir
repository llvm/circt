// RUN: circt-opt --pass-pipeline='builtin.module(firrtl.circuit(firrtl-specialize-layers))' %s | FileCheck %s

//===----------------------------------------------------------------------===//
// The Basics
//===----------------------------------------------------------------------===//

// Should not crash with no specialization.
// CHECK-LABEL: firrtl.circuit "Identity"
firrtl.circuit "Identity" {
  firrtl.extmodule @Identity()
}

// Should remove the enable and disable attributes from the circuit.
// CHECK-LABEL: firrtl.circuit "NoLayers"
firrtl.circuit "NoLayers" attributes {
    // CHECK-NOT: enable_layers = [],
    enable_layers = [],
    // CHECK-NOT: disable_layers = []
    disable_layers = []
  } {
  firrtl.extmodule @NoLayers()
}

// Should allow double disabling layers.
firrtl.circuit "DisableTwice" attributes {
    disable_layers = [@A, @A]
  } {
  firrtl.layer @A bind { }
  firrtl.extmodule @DisableTwice()
}

// CHECK-LABEL: firrtl.circuit "DefaultSpecialization"
firrtl.circuit "DefaultSpecialization" attributes {
    // This option does not get removed - it is idempotent to leave it.
    // CHECK: default_layer_specialization = #firrtl<layerspecialization enable>
    default_layer_specialization = #firrtl<layerspecialization enable>
  } {
  firrtl.extmodule @DefaultSpecialization()
}

//===----------------------------------------------------------------------===//
// Specialize Layer Declarations
//===----------------------------------------------------------------------===//

// CHECK-LABEL: firrtl.circuit "LayerEnableA"
firrtl.circuit "LayerEnableA" attributes {
    enable_layers = [@A]
  } {
  // CHECK:      firrtl.layer @A_B bind {
  // CHECK-NEXT:   firrtl.layer @C bind {
  // CHECK-NEXT:   }
  // CHECK-NEXT: }
  firrtl.layer @A bind {
    firrtl.layer @B bind {
      firrtl.layer @C bind { }
    }
  }
  firrtl.extmodule @LayerEnableA()
}

// CHECK-LABEL: firrtl.circuit "LayerEnableB"
firrtl.circuit "LayerEnableB" attributes {
    enable_layers = [@A::@B]
  } {
  // CHECK:      firrtl.layer @A bind {
  // CHECK-NEXT:   firrtl.layer @B_C bind {
  // CHECK-NEXT:   }
  // CHECK-NEXT: }
  firrtl.layer @A bind {
    firrtl.layer @B bind {
      firrtl.layer @C bind { }
    }
  }
  firrtl.extmodule @LayerEnableB()
}

// CHECK-LABEL: firrtl.circuit "LayerEnableBoth"
firrtl.circuit "LayerEnableBoth" attributes {
    enable_layers = [@A, @A::@B]
  } {
  // CHECK:      firrtl.layer @A_B_C bind {
  // CHECK-NEXT: }
  firrtl.layer @A bind {
    firrtl.layer @B bind {
      firrtl.layer @C bind { }
    }
  }
  firrtl.extmodule @LayerEnableBoth()
}

// CHECK-LABEL: firrtl.circuit "LayerDisableB"
firrtl.circuit "LayerDisableB" attributes {
    disable_layers = [@A::@B]
  } {
  // CHECK:      firrtl.layer @A bind {
  // CHECK-NEXT: }
  firrtl.layer @A bind {
    firrtl.layer @B bind {
      firrtl.layer @C bind { }
    }
  }
  firrtl.extmodule @LayerDisableB()
}

// CHECK-LABEL: firrtl.circuit "LayerDisableInARow"
firrtl.circuit "LayerDisableInARow" attributes {
    disable_layers = [@A, @B, @C]
  } {
  // CHECK-NOT: firrtl.layer @A bind { }
  firrtl.layer @A bind { }
  // CHECK-NOT: firrtl.layer @B bind { }
  firrtl.layer @B bind { }
  // CHECK-NOT: firrtl.layer @C bind { }
  firrtl.layer @C bind { }
  firrtl.extmodule @LayerDisableInARow()
}

// CHECK:     firrtl.circuit "LayerblockEnableNestedChildren"
// CHECK-NOT:   firrtl.layer
firrtl.circuit "LayerblockEnableNestedChildren" attributes {
  enable_layers = [@A, @A::@B, @A::@C]
} {
  firrtl.layer @A bind {
    firrtl.layer @B bind {
    }
    firrtl.layer @C bind {
    }
  }
  firrtl.module @LayerblockEnableNestedChildren() {
  }
}

//===----------------------------------------------------------------------===//
// LayerBlock Specialization
//===----------------------------------------------------------------------===//

// CHECK-LABEL: firrtl.circuit "LayerblockEnableA"
firrtl.circuit "LayerblockEnableA" attributes {
    enable_layers = [@A]
  } {
  firrtl.layer @A bind {
    firrtl.layer @B bind {
      firrtl.layer @C bind { }
    }
  }
  firrtl.module @LayerblockEnableA() {
    // CHECK:      firrtl.layerblock @A_B {
    // CHECK-NEXT:   firrtl.layerblock @A_B::@C {
    // CHECK-NEXT:   }
    // CHECK-NEXT: }
    firrtl.layerblock @A {
      firrtl.layerblock @A::@B {
        firrtl.layerblock @A::@B::@C { }
      }
    }
  }
}

// CHECK-LABEL: firrtl.circuit "LayerblockEnableB"
firrtl.circuit "LayerblockEnableB" attributes {
    enable_layers = [@A::@B]
  } {
  firrtl.layer @A bind {
    firrtl.layer @B bind {
      firrtl.layer @C bind { }
    }
  }
  firrtl.module @LayerblockEnableB() {
    // CHECK:      firrtl.layerblock @A {
    // CHECK-NEXT:   firrtl.layerblock @A::@B_C {
    // CHECK-NEXT:   }
    // CHECK-NEXT: }
    firrtl.layerblock @A {
      firrtl.layerblock @A::@B {
        firrtl.layerblock @A::@B::@C { }
      }
    }
  }
}

// CHECK-LABEL: firrtl.circuit "LayerblockEnableBoth"
firrtl.circuit "LayerblockEnableBoth" attributes {
    enable_layers = [@A, @A::@B]
  } {
  firrtl.layer @A bind {
    firrtl.layer @B bind {
      firrtl.layer @C bind { }
    }
  }
  firrtl.module @LayerblockEnableBoth() {
    // CHECK:      firrtl.layerblock @A_B_C {
    // CHECK-NEXT: }
    firrtl.layerblock @A {
      firrtl.layerblock @A::@B {
        firrtl.layerblock @A::@B::@C { }
      }
    }
  }
}

// CHECK-LABEL: firrtl.circuit "LayerblockDisableB"
firrtl.circuit "LayerblockDisableB" attributes {
    disable_layers = [@A::@B]
  } {
  firrtl.layer @A bind {
    firrtl.layer @B bind {
      firrtl.layer @C bind { }
    }
  }
  firrtl.module @LayerblockDisableB() {
    // CHECK:      firrtl.layerblock @A {
    // CHECK-NEXT: }
    firrtl.layerblock @A {
      firrtl.layerblock @A::@B {
        firrtl.layerblock @A::@B::@C { }
      }
    }
  }
}

//===----------------------------------------------------------------------===//
// Default Specialization
//===----------------------------------------------------------------------===//

// CHECK-LABEL: firrtl.circuit "LayerDefaultEnable"
firrtl.circuit "LayerDefaultEnable" attributes {
    disable_layers = [@A],
    default_layer_specialization = #firrtl<layerspecialization enable>
  } {
  firrtl.layer @A bind { }
  firrtl.layer @B bind { }
  firrtl.module @LayerDefaultEnable() {
    // CHECK-NOT: firrtl.layerblock @A
    // CHECK-NOT: w0 = firrtl.wire : !firrtl.uint<1>
    firrtl.layerblock @A {
      %w0 = firrtl.wire : !firrtl.uint<1>
    }
    // CHECK-NOT: firrtl.layerblock @C
    // CHECK:     w1 = firrtl.wire : !firrtl.uint<1>
    firrtl.layerblock @B {
      %w1 = firrtl.wire : !firrtl.uint<1>
    }
  }
}

// CHECK-LABEL: firrtl.circuit "LayerDefaultDisable"
firrtl.circuit "LayerDefaultDisable" attributes {
    enable_layers = [@A],
    default_layer_specialization = #firrtl<layerspecialization disable>
  } {
  firrtl.layer @A bind { }
  firrtl.layer @B bind { }
  firrtl.module @LayerDefaultDisable() {
    // CHECK-NOT: firrtl.layerblock @A
    // CHECK: w0 = firrtl.wire : !firrtl.uint<1>
    firrtl.layerblock @A {
      %w0 = firrtl.wire : !firrtl.uint<1>
    }
    // CHECK-NOT: firrtl.layerblock @C
    // CHECK-NOT: w1 = firrtl.wire : !firrtl.uint<1>
    firrtl.layerblock @B {
      %w1 = firrtl.wire : !firrtl.uint<1>
    }
  }
}

//===----------------------------------------------------------------------===//
// Module Enable Layers
//===----------------------------------------------------------------------===//

// CHECK-LABEL: firrtl.circuit "EnableLayerB"
firrtl.circuit "EnableLayerB"
attributes {
    enable_layers = [@A::@B]
  } {
  firrtl.layer @A bind {
    firrtl.layer @B bind {
      firrtl.layer @C bind { }
    }
  }
  firrtl.option @Option {
    firrtl.option_case @Option
  }

  firrtl.module public @EnableLayerB() {
    firrtl.layerblock @A {
      firrtl.layerblock @A::@B {
        firrtl.layerblock @A::@B::@C {
          // CHECK: firrtl.instance instance {layers = [@A, @A, @A::@B_C]} @ExtModule()
          firrtl.instance instance {layers = [@A, @A::@B, @A::@B::@C]} @ExtModule()
          // CHECK: firrtl.instance_choice instance_choice
          // CHECK-SAME: {layers = [@A, @A, @A::@B_C]} @ExtModule
          firrtl.instance_choice instance_choice
            {layers = [@A, @A::@B, @A::@B::@C]}
            @ExtModule alternatives @Option { @Option -> @ExtModule } ()
        }
      }
    }
  }

  // CHECK: firrtl.extmodule @ExtModule() attributes {
  // CHECK-SAME: knownLayers = [@A, @A, @A::@B_C]
  // CHECK-SAME: layers      = [@A, @A, @A::@B_C]
  firrtl.extmodule @ExtModule() attributes {
    knownLayers = [@A, @A::@B, @A::@B::@C],
    layers      = [@A, @A::@B, @A::@B::@C]
  }
}

// CHECK-LABEL: firrtl.circuit "DisableLayerA"
firrtl.circuit "DisableLayerA" attributes {
    disable_layers = [@A]
  } {
  firrtl.layer @A bind { }
  // CHECK-NOT: firrtl.extmodule @Test0()
  firrtl.extmodule @Test0() attributes {knownLayers = [@A], layers = [@A]}
  // CHECK-NOT: firrtl.extmodule @Test1() attributes {knownLayers = [@A], layers = [@A]}
  firrtl.extmodule @Test1() attributes {knownLayers = [@A], layers = [@A]}
  // CHECK-NOT: firrtl.extmodule @Test2() attributes {knownLayers = [@A], layers = [@A]}
  firrtl.extmodule @Test2() attributes {knownLayers = [@A], layers = [@A]}

  // Top level module, which can't be deleted by the pass.
  firrtl.extmodule @DisableLayerA()
}

//===----------------------------------------------------------------------===//
// Probe Types
//===----------------------------------------------------------------------===//

// CHECK-LABEL: firrtl.circuit "ProbeOpsEnableA"
firrtl.circuit "ProbeOpsEnableA" attributes {
  enable_layers = [@A]
} {
  firrtl.layer @A bind { }
  firrtl.option @Option { firrtl.option_case @Option }
  // CHECK: firrtl.module @ProbeOpsEnableA(out %out: !firrtl.probe<uint<1>>)
  firrtl.module @ProbeOpsEnableA(out %out : !firrtl.probe<uint<1>, @A>) {
    // CHECK: %w1 = firrtl.wire : !firrtl.probe<bundle<a: uint<1>>>
    %w1 = firrtl.wire : !firrtl.probe<bundle<a: uint<1>>, @A>
    // CHECK: %0 = firrtl.ref.sub %w1[0] : !firrtl.probe<bundle<a: uint<1>>>
    %1 = firrtl.ref.sub %w1[0] : !firrtl.probe<bundle<a: uint<1>>, @A>
    // CHECK: %1 = firrtl.ref.cast %0 : (!firrtl.probe<uint<1>>) -> !firrtl.probe<uint<1>, @A_B>
    %2 = firrtl.ref.cast %1 : (!firrtl.probe<uint<1>, @A>) -> !firrtl.probe<uint<1>, @A::@B>
    // CHECK: firrtl.ref.define %out, %0 : !firrtl.probe<uint<1>>
    firrtl.ref.define %out, %1 : !firrtl.probe<uint<1>, @A>
    %c0_ui1 = firrtl.constant 0 : !firrtl.uint<1>
    // CHECK: firrtl.when %c0_ui1 : !firrtl.uint<1> {
    firrtl.when %c0_ui1 : !firrtl.uint<1> {
      // CHECK: %w2 = firrtl.wire : !firrtl.probe<uint<1>>
      %w2 = firrtl.wire : !firrtl.probe<uint<1>, @A>
    } else {
      // CHECK: %w3 = firrtl.wire : !firrtl.probe<uint<1>>
      %w3 = firrtl.wire : !firrtl.probe<uint<1>, @A>
    }

    %enum = firrtl.wire : !firrtl.enum<a: uint<8>>
    // CHECK: firrtl.match %enum : !firrtl.enum<a: uint<8>> {
    firrtl.match %enum : !firrtl.enum<a: uint<8>> {
      case a(%arg0) {
        // CHECK: %w4 = firrtl.wire : !firrtl.probe<uint<1>>
        %w4 = firrtl.wire : !firrtl.probe<uint<1>, @A>
      }
    }

    // CHECK: firrtl.instance instance @ExtModule(out out: !firrtl.probe<uint<1>>)
    firrtl.instance instance @ExtModule(out out : !firrtl.probe<uint<1>, @A>)

    // CHECK: firrtl.instance_choice instance_choice @ExtModule
    // CHECK-SAME: alternatives @Option { @Option -> @ExtModule }
    // CHECK-SAME: (out out: !firrtl.probe<uint<1>>)
    firrtl.instance_choice instance_choice @ExtModule
      alternatives @Option { @Option -> @ExtModule }
      (out out : !firrtl.probe<uint<1>, @A>)
  }

  firrtl.extmodule @ExtModule(out out : !firrtl.probe<uint<1>, @A>) attributes {knownLayers=[@A]}
}

// CHECK-LABEL: firrtl.circuit "ProbeOpsDisableA"
firrtl.circuit "ProbeOpsDisableA" attributes {
  disable_layers = [@A]
} {
  firrtl.layer @A bind { }
  firrtl.option @Option { firrtl.option_case @Option }
  // CHECK: firrtl.module @ProbeOpsDisableA()
  firrtl.module @ProbeOpsDisableA(out %out : !firrtl.probe<uint<1>, @A>) {
    // CHECK-NOT: firrtl.wire
    %w1 = firrtl.wire : !firrtl.probe<bundle<a: uint<1>>, @A>
    // CHECK-NOT: firrtl.ref.sub
    %1 = firrtl.ref.sub %w1[0] : !firrtl.probe<bundle<a: uint<1>>, @A>
    // CHECK-NOT: firrtl.ref.cast
    %2 = firrtl.ref.cast %1 : (!firrtl.probe<uint<1>, @A>) -> !firrtl.probe<uint<1>, @A::@B>
    // CHECK-NOT: firrtl.ref.define
    firrtl.ref.define %out, %1 : !firrtl.probe<uint<1>, @A>
    %c0_ui1 = firrtl.constant 0 : !firrtl.uint<1>
    // CHECK: firrtl.when %c0_ui1 : !firrtl.uint<1> {
    firrtl.when %c0_ui1 : !firrtl.uint<1> {
      // CHECK-NOT: %w2 = firrtl.wire
      %w2 = firrtl.wire : !firrtl.probe<uint<1>, @A>
    } else {
      // CHECK-NOT: %w3 = firrtl.wire
      %w3 = firrtl.wire : !firrtl.probe<uint<1>, @A>
    }

    %enum = firrtl.wire : !firrtl.enum<a: uint<8>>
    // CHECK: firrtl.match %enum : !firrtl.enum<a: uint<8>> {
    firrtl.match %enum : !firrtl.enum<a: uint<8>> {
      case a(%arg0) {
        // CHECK-NOT: %w4 = firrtl.wire : !firrtl.probe<uint<1>, @A>
        %w4 = firrtl.wire : !firrtl.probe<uint<1>, @A>
      }
    }

    // CHECK: firrtl.instance instance @ExtModule()
    firrtl.instance instance @ExtModule(out out : !firrtl.probe<uint<1>, @A>)
    // CHECK: firrtl.instance_choice instance_choice @ExtModule
    // CHECK-SAME: alternatives @Option { @Option -> @ExtModule }
    // CHECK-SAME: ()
    firrtl.instance_choice instance_choice @ExtModule
      alternatives @Option { @Option -> @ExtModule }
      (out out : !firrtl.probe<uint<1>, @A>)
  }

  // CHECK: firrtl.extmodule @ExtModule()
  firrtl.extmodule @ExtModule(out out : !firrtl.probe<uint<1>, @A>) attributes {knownLayers=[@A]}
}

//===----------------------------------------------------------------------===//
// Hierarchical Paths
//===----------------------------------------------------------------------===//

// All hierarchical paths which traverse a deleted instance should be removed.

// CHECK-LABEL: firrtl.circuit "HierPathDelete"
firrtl.circuit "HierPathDelete" attributes {
  disable_layers = [@Layer]
} {
  firrtl.layer @Layer bind { }

  // CHECK-NOT: hw.hierpath private @hierpath1 [@HierPathDelete::@middle, @Middle::@leaf, @Leaf]
  hw.hierpath private @hierpath1 [@HierPathDelete::@middle, @Middle::@leaf, @Leaf]
  // CHECK-NOT: hw.hierpath private @hierpath2 [@HierPathDelete::@middle, @Middle::@leaf]
  hw.hierpath private @hierpath2 [@HierPathDelete::@middle, @Middle::@leaf]
  // CHECK-NOT: hw.hierpath private @hierpath3 [@Middle::@leaf, @Leaf]
  hw.hierpath private @hierpath3 [@Middle::@leaf, @Leaf]
  // CHECK-NOT: hw.hierpath private @hierpath4 [@Middle::@leaf]
  hw.hierpath private @hierpath4 [@Middle::@leaf]

  firrtl.module @HierPathDelete() {
    firrtl.instance middle sym @middle @Middle()
  }

  firrtl.module @Middle() {
    firrtl.layerblock @Layer {
      firrtl.instance leaf sym @leaf @Leaf()
    }
  }

  firrtl.extmodule @Leaf()

  // CHECK-NOT: hw.hierpath private @DeletedPath [@Deleted]
  hw.hierpath private @DeletedPath [@Deleted]
  firrtl.extmodule @Deleted() attributes {knownLayers=[@Layer], layers = [@Layer]}
}

// CHECK-LABEL: firrtl.circuit "HierPathDelete2"
firrtl.circuit "HierPathDelete2" attributes {
  disable_layers = [@Layer]
} {
  firrtl.layer @Layer bind { }

  // CHECK-NOT: hw.hierpath private @hierpath1 [@HierPathDelete2::@middle, @Middle::@leaf, @Leaf]
  hw.hierpath private @hierpath1 [@HierPathDelete2::@middle, @Middle::@leaf, @Leaf]
  // CHECK-NOT: hw.hierpath private @hierpath2 [@HierPathDelete2::@middle, @Middle::@leaf]
  hw.hierpath private @hierpath2 [@HierPathDelete2::@middle, @Middle::@leaf]
  // CHECK-NOT: hw.hierpath private @hierpath3 [@Middle::@leaf, @Leaf]
  hw.hierpath private @hierpath3 [@Middle::@leaf, @Leaf]
  // CHECK-NOT: hw.hierpath private @hierpath4 [@Middle::@leaf]
  hw.hierpath private @hierpath4 [@Middle::@leaf]

  firrtl.module @HierPathDelete2() {
    firrtl.layerblock @Layer {
      firrtl.instance middle sym @middle @Middle()
    }
  }

  firrtl.module @Middle() {
    firrtl.layerblock @Layer {
      firrtl.instance leaf sym @leaf @Leaf()
    }
  }

  firrtl.extmodule @Leaf()

  // CHECK-NOT: hw.hierpath private @DeletedPath [@Deleted]
  hw.hierpath private @DeletedPath [@Deleted]
  firrtl.extmodule @Deleted() attributes {knownLayers=[@Layer], layers = [@Layer]}
}


//===----------------------------------------------------------------------===//
// Annotations
//===----------------------------------------------------------------------===//

// All non-local annotations which have a path which no longer exists after
// specialization occurs should be removed.
// CHECK-LABEL: firrtl.circuit "Annotations"
firrtl.circuit "Annotations" attributes {
  disable_layers = [@Layer]
} {
  firrtl.layer @Layer bind { }
  hw.hierpath private @Path [@Annotations::@leaf, @Leaf]
  firrtl.module @Annotations() {
    firrtl.layerblock @Layer {
      firrtl.instance leaf sym @leaf @Leaf(in in : !firrtl.uint<1>)
    }
  }

  // CHECK: firrtl.module @Leaf(in %in: !firrtl.uint<1>) {
  firrtl.module @Leaf(in %in : !firrtl.uint<1> [{circt.nonlocal = @Path}])
    attributes {annotations = [{circt.nonlocal = @Path}]} {

    // CHECK: %w = firrtl.wire : !firrtl.uint<1>
    %w = firrtl.wire {annotations = [{circt.nonlocal = @Path}]} : !firrtl.uint<1>

    // CHECK: %mem_w = firrtl.mem Undefined {depth = 12 : i64, name = "mem", portNames = ["w"], readLatency = 0 : i32, writeLatency = 1 : i32} : !firrtl.bundle<addr: uint<4>, en: uint<1>, clk: clock, data: uint<0>, mask: uint<1>>
    %mem_w = firrtl.mem Undefined {depth = 12 : i64, name = "mem", portAnnotations = [[{circt.nonlocal = @Path}]], portNames = ["w"], readLatency = 0 : i32, writeLatency = 1 : i32} : !firrtl.bundle<addr: uint<4>, en: uint<1>, clk: clock, data: uint<0>, mask: uint<1>>
  }
}
