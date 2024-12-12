// RUN: circt-opt -firrtl-lower-layers -split-input-file %s | FileCheck %s

firrtl.circuit "Test" {
  firrtl.module @Test() {}

  firrtl.layer @A bind {
    firrtl.layer @B bind {
      firrtl.layer @C bind {}
    }
  }
  firrtl.layer @B bind {}

  firrtl.extmodule @Foo(out o : !firrtl.probe<uint<1>, @A>)

  //===--------------------------------------------------------------------===//
  // Removal of Probe Colors
  //===--------------------------------------------------------------------===//

  // CHECK-LABEL: @ColoredPorts(out %o: !firrtl.probe<uint<1>>)
  firrtl.module @ColoredPorts(out %o: !firrtl.probe<uint<1>, @A>) {}

  // CHECK-LABEL: @ExtColoredPorts(out o: !firrtl.probe<uint<1>>)
  firrtl.extmodule @ExtColoredPorts(out o: !firrtl.probe<uint<1>, @A>)

  // CHECK-LABEL: @ColoredPortsOnInstances
  firrtl.module @ColoredPortsOnInstances() {
    // CHECK: %foo_o = firrtl.instance foo @ColoredPorts(out o: !firrtl.probe<uint<1>>)
   %foo_o = firrtl.instance foo @ColoredPorts(out o: !firrtl.probe<uint<1>, @A>)
  }

  // CHECK-LABEL: @ColoredThings
  firrtl.module @ColoredThings() {
    // CHECK: %0 = firrtl.wire : !firrtl.probe<bundle<f: uint<1>>>
    %0 = firrtl.wire : !firrtl.probe<bundle<f: uint<1>>, @A>
    // CHECK: %1 = firrtl.ref.sub %0[0] : !firrtl.probe<bundle<f: uint<1>>>
    %1 = firrtl.ref.sub %0[0] : !firrtl.probe<bundle<f: uint<1>>, @A>
    // CHECK-NOT: firrtl.ref.cast
    %2 = firrtl.ref.cast %1 : (!firrtl.probe<uint<1>, @A>) -> !firrtl.probe<uint<1>, @A::@B>
  }

    // CHECK-LABEL: @ColoredThingUnderWhen
  firrtl.module @ColoredThingUnderWhen(in %b : !firrtl.uint<1>) {
    // CHECK: firrtl.when %b : !firrtl.uint<1>
    firrtl.when %b : !firrtl.uint<1> {
      // CHECK: %0 = firrtl.wire : !firrtl.probe<bundle<f: uint<1>>>
      %0 = firrtl.wire : !firrtl.probe<bundle<f: uint<1>>, @A>
      // CHECK: %1 = firrtl.ref.sub %0[0] : !firrtl.probe<bundle<f: uint<1>>>
      %1 = firrtl.ref.sub %0[0] : !firrtl.probe<bundle<f: uint<1>>, @A>
      // CHECK-NOT: firrtl.ref.cast
      %2 = firrtl.ref.cast %1 : (!firrtl.probe<uint<1>, @A>) -> !firrtl.probe<uint<1>, @A::@B>
    }
  }

  //===--------------------------------------------------------------------===//
  // Removal of Enabled Layers
  //===--------------------------------------------------------------------===//

  // CHECK-LABEL: @EnabledLayers() {
  firrtl.module @EnabledLayers() attributes {layers = [@A]} {}

  // CHECK-LABEL: @EnabledLayersOnInstance()
  firrtl.module @EnabledLayersOnInstance() attributes {layers = [@A]} {
    // CHECK: firrtl.instance enabledLayers @EnabledLayers()
    firrtl.instance enabledLayers {layers = [@A]} @EnabledLayers()
  }

  //===--------------------------------------------------------------------===//
  // Removal of Layerblocks and Layers
  //===--------------------------------------------------------------------===//

  // CHECK-NOT: firrtl.layer @GoodbyeCruelWorld
  firrtl.layer @GoodbyeCruelWorld bind {}

  // CHECK-LABEL @WithLayerBlock
  firrtl.module @WithLayerBlock() {
    // CHECK-NOT firrtl.layerblock @GoodbyeCruelWorld
    firrtl.layerblock @GoodbyeCruelWorld {
    }
  }

  //===--------------------------------------------------------------------===//
  // Capture
  //===--------------------------------------------------------------------===//

  // CHECK: firrtl.module private @[[A:.+]](in %[[x:.+]]: !firrtl.uint<1>, in %[[y:.+]]: !firrtl.uint<1>)
  // CHECK:   %0 = firrtl.add %[[x]], %[[y]] : (!firrtl.uint<1>, !firrtl.uint<1>) -> !firrtl.uint<2>
  // CHECK: }
  // CHECK: firrtl.module @CaptureHardware() {
  // CHECK:   %c0_ui1 = firrtl.constant 0 : !firrtl.uint<1>
  // CHECK:   %c1_ui1 = firrtl.constant 1 : !firrtl.uint<1>
  // CHECK:   %[[p:.+]], %[[q:.+]] = firrtl.instance {{.+}} {lowerToBind, output_file = #hw.output_file<"layers-Test-A.sv", excludeFromFileList>} @[[A]]
  // CHECK:   firrtl.matchingconnect %[[q]], %c1_ui1 : !firrtl.uint<1>
  // CHECK:   firrtl.matchingconnect %[[p]], %c0_ui1 : !firrtl.uint<1>
  // CHECK: }
  firrtl.module @CaptureHardware() {
    %c0_ui1 = firrtl.constant 0 : !firrtl.uint<1>
    %c1_ui1 = firrtl.constant 1 : !firrtl.uint<1>
    firrtl.layerblock @A {
      %0 = firrtl.add %c0_ui1, %c1_ui1 : (!firrtl.uint<1>, !firrtl.uint<1>) -> !firrtl.uint<2>
    }
  }

  // CHECK: firrtl.module private @[[A:.+]](in %[[p:.+]]: !firrtl.uint<1>) {
  // CHECK:   %x = firrtl.node %[[p]] : !firrtl.uint<1>
  // CHECK: }
  // CHECK: firrtl.module @CapturePort(in %in: !firrtl.uint<1>) {
  // CHECK:   %[[p:.+]] = firrtl.instance {{.+}} {lowerToBind, output_file = #hw.output_file<"layers-Test-A.sv", excludeFromFileList>} @[[A]]
  // CHECK:   firrtl.matchingconnect %[[p]], %in : !firrtl.uint<1>
  // CHECK: }
  firrtl.module @CapturePort(in %in: !firrtl.uint<1>){
    firrtl.layerblock @A {
      %x = firrtl.node %in : !firrtl.uint<1>
    }
  }

  // CHECK: firrtl.module private @[[A:.+]](in %[[p:.+]]: !firrtl.uint<1>)
  // CHECK:   %w = firrtl.wire : !firrtl.uint<1>
  // CHECK:   firrtl.connect %w, %[[p]] : !firrtl.uint<1>
  // CHECK: }
  // CHECK: firrtl.module @CaptureHardwareViaConnect() {
  // CHECK:   %c0_ui1 = firrtl.constant 0 : !firrtl.uint<1>
  // CHECK:   %[[p:.+]] = firrtl.instance {{.+}} {lowerToBind, output_file = #hw.output_file<"layers-Test-A.sv", excludeFromFileList>} @[[A]]
  // CHECK:   firrtl.matchingconnect %[[p]], %c0_ui1 : !firrtl.uint<1>
  // CHECK: }
  firrtl.module @CaptureHardwareViaConnect() {
    %c0_ui1 = firrtl.constant 0 : !firrtl.uint<1>
    firrtl.layerblock @A {
      %w = firrtl.wire : !firrtl.uint<1>
      firrtl.connect %w, %c0_ui1 : !firrtl.uint<1>, !firrtl.uint<1>
    }
  }

  // CHECK: firrtl.module private @[[A:.+]](in %[[p:.+]]: !firrtl.uint<1>)
  // CHECK:   %0 = firrtl.ref.send %[[p]] : !firrtl.uint<1>
  // CHECK:   %1 = firrtl.ref.resolve %0 : !firrtl.probe<uint<1>>
  // CHECK: }
  // CHECK: firrtl.module @CaptureProbeSrc() {
  // CHECK:   %c0_ui1 = firrtl.constant 0 : !firrtl.uint<1>
  // CHECK:   %w = firrtl.wire : !firrtl.uint<1>
  // CHECK:   %0 = firrtl.ref.send %w : !firrtl.uint<1>
  // CHECK:   %[[p:.+]] = firrtl.instance {{.+}} {lowerToBind, output_file = #hw.output_file<"layers-Test-A.sv", excludeFromFileList>} @[[A]]
  // CHECK:   %1 = firrtl.ref.resolve %0 : !firrtl.probe<uint<1>>
  // CHECK:   firrtl.matchingconnect %[[p]], %1 : !firrtl.uint<1>
  // CHECK: }
  firrtl.module @CaptureProbeSrc() {
    %c0_ui1 = firrtl.constant 0 : !firrtl.uint<1>
    %w = firrtl.wire : !firrtl.uint<1>
    %r = firrtl.ref.send %w : !firrtl.uint<1>
    firrtl.layerblock @A {
      firrtl.ref.resolve %r : !firrtl.probe<uint<1>>
    }
  }

  // CHECK: firrtl.module private @[[B:.+]](in %[[p:.+]]: !firrtl.uint<1>, in %[[q:.+]]: !firrtl.uint<1>)
  // CHECK:   %0 = firrtl.add %[[p]], %[[q]] : (!firrtl.uint<1>, !firrtl.uint<1>) -> !firrtl.uint<2>
  // CHECK: }
  // CHECK: firrtl.module private @[[A:.+]](out %[[p:.+]]: !firrtl.probe<uint<1>>, out %[[q:.+]]: !firrtl.probe<uint<1>>) attributes {
  // CHECK:   %c0_ui1 = firrtl.constant 0 : !firrtl.uint<1>
  // CHECK:   %0 = firrtl.ref.send %c0_ui1 : !firrtl.uint<1>
  // CHECK:   firrtl.ref.define %[[q]], %0 : !firrtl.probe<uint<1>>
  // CHECK:   %c0_ui1_1 = firrtl.constant 0 : !firrtl.uint<1>
  // CHECK:   %1 = firrtl.ref.send %c0_ui1_1 : !firrtl.uint<1>
  // CHECK:   firrtl.ref.define %[[p]], %1 : !firrtl.probe<uint<1>>
  // CHECK: }
  // CHECK: firrtl.module @NestedCaptureHardware() {
  // CHECK:   %[[b1:.+]], %[[b2:.+]] = firrtl.instance {{.+}} {lowerToBind, output_file = #hw.output_file<"layers-Test-A-B.sv", excludeFromFileList>} @[[B]]
  // CHECK:   %[[a1:.+]], %[[a2:.+]] = firrtl.instance {{.+}} {lowerToBind, output_file = #hw.output_file<"layers-Test-A.sv", excludeFromFileList>} @[[A]]
  // CHECK:   %0 = firrtl.ref.resolve %[[a2]] : !firrtl.probe<uint<1>>
  // CHECK:   firrtl.matchingconnect %[[b1]], %0 : !firrtl.uint<1>
  // CHECK:   %1 = firrtl.ref.resolve %[[a1]] : !firrtl.probe<uint<1>>
  // CHECK:   firrtl.matchingconnect %[[b2]], %1 : !firrtl.uint<1>
  // CHECK: }
  firrtl.module @NestedCaptureHardware() {
    firrtl.layerblock @A {
      %c0_ui1 = firrtl.constant 0 : !firrtl.uint<1>
      %c1_ui1 = firrtl.constant 0 : !firrtl.uint<1>
      firrtl.layerblock @A::@B {
        %0 = firrtl.add %c0_ui1, %c1_ui1 : (!firrtl.uint<1>, !firrtl.uint<1>) -> !firrtl.uint<2>
      }
    }
  }

  // CHECK: firrtl.module private @[[A:.+]](in %[[p:.+]]: !firrtl.uint<1>)
  // CHECK:   %c1_ui1 = firrtl.constant 1 : !firrtl.uint<1>
  // CHECK:   firrtl.when %[[p]] : !firrtl.uint<1> {
  // CHECK:     %0 = firrtl.add %[[p]], %c1_ui1 : (!firrtl.uint<1>, !firrtl.uint<1>) -> !firrtl.uint<2>
  // CHECK:   }
  // CHECK: }
  // CHECK: firrtl.module @WhenUnderLayer() {
  // CHECK:   %c0_ui1 = firrtl.constant 0 : !firrtl.uint<1>
  // CHECK:   %[[p:.+]] = firrtl.instance {{.+}} {lowerToBind, output_file = #hw.output_file<"layers-Test-A.sv", excludeFromFileList>} @[[A]]
  // CHECK:   firrtl.matchingconnect %[[p]], %c0_ui1 : !firrtl.uint<1>
  // CHECK: }
  firrtl.module @WhenUnderLayer() {
    %c0_ui1 = firrtl.constant 0 : !firrtl.uint<1>
    firrtl.layerblock @A {
      %c1_ui1 = firrtl.constant 1 : !firrtl.uint<1>
      firrtl.when %c0_ui1 : !firrtl.uint<1> {
        %0 = firrtl.add %c0_ui1, %c1_ui1 : (!firrtl.uint<1>, !firrtl.uint<1>) -> !firrtl.uint<2>
      }
    }
  }

  // Test that subfield, subindex, and subaccess are moved out of layerblocks to
  // avoid capturing non-passive types.
  //
  // CHECK:      firrtl.module private @[[SubOpsInLayerBlock_A:[A-Za-z0-9_]+]]
  // CHECK-SAME:   in %[[port:[A-Za-z0-9_]+]]: !firrtl.uint<1>
  // CHECK-NEXT:   firrtl.node %[[port]]
  // CHECK-NEXT: }
  // CHECK:      firrtl.module @SubOpsInLayerBlock
  // CHECK-NEXT:   firrtl.subaccess
  // CHECK-NEXT:   firrtl.subindex
  // CHECK-NEXT:   firrtl.subfield
  firrtl.module @SubOpsInLayerBlock(
    in %a: !firrtl.vector<vector<bundle<a: uint<1>, b flip: uint<2>>, 2>, 2>,
    in %b: !firrtl.uint<1>
  ) {
    firrtl.layerblock @A {
      %0 = firrtl.subaccess %a[%b] : !firrtl.vector<vector<bundle<a: uint<1>, b flip: uint<2>>, 2>, 2>, !firrtl.uint<1>
      %1 = firrtl.subindex %0[0] : !firrtl.vector<bundle<a: uint<1>, b flip: uint<2>>, 2>
      %2 = firrtl.subfield %1[a] : !firrtl.bundle<a: uint<1>, b flip: uint<2>>
      %3 = firrtl.node %2 : !firrtl.uint<1>
    }
  }

  // CHECK:      firrtl.module private @CaptureInWhen_A(
  // CHECK-SAME:   in %a: !firrtl.uint<1>
  // CHECK-SAME:   in %cond: !firrtl.uint<1>
  // CHECK-SAME: )

  // CHECK:      firrtl.module @CaptureInWhen(
  // CHECK:        %a_a, %a_cond = firrtl.instance a
  // CHECK-NEXT:   firrtl.matchingconnect %a_cond, %cond :
  // CHECK-NEXT:   firrtl.matchingconnect %a_a, %a :
  firrtl.module @CaptureInWhen(in %cond: !firrtl.uint<1>) {
    %a = firrtl.wire : !firrtl.uint<1>
    firrtl.layerblock @A {
      firrtl.when %cond : !firrtl.uint<1> {
        %b = firrtl.node %a : !firrtl.uint<1>
      }
    }
  }

  //===--------------------------------------------------------------------===//
  // Connecting/Defining Refs
  //===--------------------------------------------------------------------===//

  // Src and Dst Outside Layerblock.
  //
  // CHECK: firrtl.module private @[[A:.+]]() {
  // CHECK: }
  // CHECK: firrtl.module @SrcDstOutside() {
  // CHECK:   %0 = firrtl.wire : !firrtl.probe<uint<1>>
  // CHECK:   %1 = firrtl.wire : !firrtl.probe<uint<1>>
  // CHECK:   firrtl.ref.define %1, %0 : !firrtl.probe<uint<1>>
  // CHECK:   firrtl.instance {{.+}} {lowerToBind, output_file = #hw.output_file<"layers-Test-A.sv", excludeFromFileList>} @[[A:.+]]()
  // CHECK: }
  firrtl.module @SrcDstOutside() {
    %0 = firrtl.wire : !firrtl.probe<uint<1>, @A>
    %1 = firrtl.wire : !firrtl.probe<uint<1>, @A>
    firrtl.layerblock @A {
      firrtl.ref.define %1, %0 : !firrtl.probe<uint<1>, @A>
    }
  }

  // Src Outside Layerblock.
  //
  // CHECK: firrtl.module private @[[A:.+]](in %[[p:.+]]: !firrtl.uint<1>)
  // CHECK:   %0 = firrtl.ref.send %[[p]] : !firrtl.uint<1>
  // CHECK:   %1 = firrtl.wire : !firrtl.probe<uint<1>>
  // CHECK:   firrtl.ref.define %1, %0 : !firrtl.probe<uint<1>>
  // CHECK: }
  // CHECK: firrtl.module @SrcOutside() {
  // CHECK:   %0 = firrtl.wire : !firrtl.probe<uint<1>>
  // CHECK:   %[[p:.+]] = firrtl.instance {{.+}} {lowerToBind, output_file = #hw.output_file<"layers-Test-A.sv", excludeFromFileList>} @[[A]]
  // CHECK:   %1 = firrtl.ref.resolve %0 : !firrtl.probe<uint<1>>
  // CHECK:   firrtl.matchingconnect %[[p]], %1 : !firrtl.uint<1>
  // CHECK: }
  firrtl.module @SrcOutside() {
    %0 = firrtl.wire : !firrtl.probe<uint<1>, @A>
    firrtl.layerblock @A {
      %1 = firrtl.wire : !firrtl.probe<uint<1>, @A>
      firrtl.ref.define %1, %0 : !firrtl.probe<uint<1>, @A>
    }
  }

  // Dst Outside Layerblock.
  //
  // CHECK: firrtl.module private @[[A:.+]](out %[[p:.+]]: !firrtl.probe<uint<1>>)
  // CHECK:   %0 = firrtl.wire : !firrtl.probe<uint<1>>
  // CHECK:   firrtl.ref.define %[[p]], %0 : !firrtl.probe<uint<1>>
  // CHECK: }
  // CHECK: firrtl.module @DestOutside() {
  // CHECK:   %0 = firrtl.wire : !firrtl.probe<uint<1>>
  // CHECK:   %[[p:.+]] = firrtl.instance {{.+}} {lowerToBind, output_file = #hw.output_file<"layers-Test-A.sv", excludeFromFileList>} @[[A]]
  // CHECK:   firrtl.ref.define %0, %[[p]] : !firrtl.probe<uint<1>>
  // CHECK: }
  firrtl.module @DestOutside() {
    %0 = firrtl.wire : !firrtl.probe<uint<1>, @A>
    firrtl.layerblock @A {
      %1 = firrtl.wire : !firrtl.probe<uint<1>, @A>
      firrtl.ref.define %0, %1 : !firrtl.probe<uint<1>, @A>
    }
  }

  // Src and Dst Inside Layerblock.
  //
  // CHECK: firrtl.module private @[[A:.+]]() {
  // CHECK:   %0 = firrtl.wire : !firrtl.probe<uint<1>>
  // CHECK:   %1 = firrtl.wire : !firrtl.probe<uint<1>>
  // CHECK:   firrtl.ref.define %1, %0 : !firrtl.probe<uint<1>>
  // CHECK: }
  // CHECK: firrtl.module @SrcDstInside() {
  // CHECK:   firrtl.instance {{.+}} {lowerToBind, output_file = #hw.output_file<"layers-Test-A.sv", excludeFromFileList>} @[[A]]()
  // CHECK: }
  firrtl.module @SrcDstInside() {
    firrtl.layerblock @A {
      %0 = firrtl.wire : !firrtl.probe<uint<1>, @A>
      %1 = firrtl.wire : !firrtl.probe<uint<1>, @A>
      firrtl.ref.define %1, %0 : !firrtl.probe<uint<1>, @A>
    }
  }

  //===--------------------------------------------------------------------===//
  // Resolving Colored Probes
  //===--------------------------------------------------------------------===//

  // CHECK: firrtl.module private @[[A:.+]](in %[[p:.+]]: !firrtl.uint<1>) {
  // CHECK:   %0 = firrtl.ref.send %[[p]] : !firrtl.uint<1>
  // CHECK:   %1 = firrtl.ref.resolve %0 : !firrtl.probe<uint<1>>
  // CHECK: }
  // CHECK: firrtl.module @ResolveColoredRefUnderLayerBlock() {
  // CHECK:   %w = firrtl.wire : !firrtl.probe<uint<1>>
  // CHECK:   %[[p:.+]] = firrtl.instance {{.+}} {lowerToBind, output_file = #hw.output_file<"layers-Test-A.sv", excludeFromFileList>} @[[A]]
  // CHECK:   %0 = firrtl.ref.resolve %w : !firrtl.probe<uint<1>>
  // CHECK:   firrtl.matchingconnect %[[p]], %0 : !firrtl.uint<1>
  // CHECK: }
  firrtl.module @ResolveColoredRefUnderLayerBlock() {
    %w = firrtl.wire : !firrtl.probe<uint<1>, @A>
    firrtl.layerblock @A {
      %0 = firrtl.ref.resolve %w : !firrtl.probe<uint<1>, @A>
    }
  }

  // CHECK: firrtl.module @ResolveColoredRefUnderEnabledLayer() {
  // CHECK:   %w = firrtl.wire : !firrtl.probe<uint<1>>
  // CHECK:   %0 = firrtl.ref.resolve %w : !firrtl.probe<uint<1>>
  // CHECK: }
  firrtl.module @ResolveColoredRefUnderEnabledLayer() attributes {layers=[@A]} {
    %w = firrtl.wire : !firrtl.probe<uint<1>, @A>
    %0 = firrtl.ref.resolve %w : !firrtl.probe<uint<1>, @A>
  }

  // CHECK: firrtl.module private @[[A:.+]](in %[[p:.+]]: !firrtl.uint<1>) {
  // CHECK:   %0 = firrtl.ref.send %[[p]] : !firrtl.uint<1>
  // CHECK:   %1 = firrtl.ref.resolve %0 : !firrtl.probe<uint<1>>
  // CHECK: }
  // CHECK: firrtl.module @ResolveColoredRefPortUnderLayerBlock1() {
  // CHECK:   %foo_o = firrtl.instance foo @Foo(out o: !firrtl.probe<uint<1>>)
  // CHECK:   %[[p:.+]] = firrtl.instance {{.+}} {lowerToBind, output_file = #hw.output_file<"layers-Test-A.sv", excludeFromFileList>} @[[A]]
  // CHECK:   %0 = firrtl.ref.resolve %foo_o : !firrtl.probe<uint<1>>
  // CHECK:   firrtl.matchingconnect %[[p]], %0 : !firrtl.uint<1>
  // CHECK: }
  firrtl.module @ResolveColoredRefPortUnderLayerBlock1() {
    %foo_o = firrtl.instance foo @Foo(out o : !firrtl.probe<uint<1>, @A>)
    firrtl.layerblock @A {
      %x = firrtl.ref.resolve %foo_o : !firrtl.probe<uint<1>, @A>
    }
  }

  // CHECK: firrtl.module private @[[A:.+]]() {
  // CHECK:   %foo_o = firrtl.instance foo @Foo(out o: !firrtl.probe<uint<1>>)
  // CHECK:   %0 = firrtl.ref.resolve %foo_o : !firrtl.probe<uint<1>>
  // CHECK: }
  // CHECK: firrtl.module @ResolveColoredRefPortUnderLayerBlock2() {
  // CHECK:   firrtl.instance {{.+}} {lowerToBind, output_file = #hw.output_file<"layers-Test-A.sv", excludeFromFileList>} @[[A]]()
  // CHECK: }
  firrtl.module @ResolveColoredRefPortUnderLayerBlock2() {
    firrtl.layerblock @A {
      %foo_o = firrtl.instance foo @Foo(out o : !firrtl.probe<uint<1>, @A>)
      %x = firrtl.ref.resolve %foo_o : !firrtl.probe<uint<1>, @A>
    }
  }

  // CHECK: firrtl.module @ResolveColoredRefPortUnderEnabledLayer() {
  // CHECK:   %foo_o = firrtl.instance foo @Foo(out o: !firrtl.probe<uint<1>>)
  // CHECK:   %0 = firrtl.ref.resolve %foo_o : !firrtl.probe<uint<1>>
  // CHECK: }
  firrtl.module @ResolveColoredRefPortUnderEnabledLayer() attributes {layers=[@A]} {
    %foo_o = firrtl.instance foo @Foo(out o : !firrtl.probe<uint<1>, @A>)
    %x = firrtl.ref.resolve %foo_o : !firrtl.probe<uint<1>, @A>
  }

  //===--------------------------------------------------------------------===//
  // Inline Layers
  //===--------------------------------------------------------------------===//

  // CHECK:      sv.macro.decl @layer_Test$Inline
  // CHECK-NEXT: sv.macro.decl @layer_Test$Inline$Inline
  // CHECK-NEXT: sv.macro.decl @layer_Test$Bound$Inline
  firrtl.layer @Inline inline {
    firrtl.layer @Inline inline {}
  }

  firrtl.layer @Bound bind {
    firrtl.layer @Inline inline {}
  }

  // CHECK:      firrtl.module private @ModuleWithInlineLayerBlocks_Bound() {
  // CHECK-NEXT:   %w3 = firrtl.wire
  // CHECK-NEXT:   sv.ifdef @layer_Test$Bound$Inline {
  // CHECK-NEXT:     %w4 = firrtl.wire
  // CHECK-NEXT:   }
  // CHECK-NEXT: }

  // CHECK-NEXT: firrtl.module @ModuleWithInlineLayerBlocks() {
  // CHECK-NEXT:   sv.ifdef @layer_Test$Inline {
  // CHECK-NEXT:     %w1 = firrtl.wire
  // CHECK-NEXT:     sv.ifdef @layer_Test$Inline$Inline {
  // CHECK-NEXT:       %w2 = firrtl.wire
  // CHECK-NEXT:     }
  // CHECK-NEXT:   }
  // CHECK-NEXT: }
  firrtl.module @ModuleWithInlineLayerBlocks() {
    firrtl.layerblock @Inline {
      %w1 = firrtl.wire : !firrtl.uint<1>
      firrtl.layerblock @Inline::@Inline {
        %w2 = firrtl.wire : !firrtl.uint<2>
      }
    }

    firrtl.layerblock @Bound {
      %w3 = firrtl.wire : !firrtl.uint<3>
      firrtl.layerblock @Bound::@Inline {
        %w4 = firrtl.wire : !firrtl.uint<4>
      }
    }
  }

}

// -----

firrtl.circuit "Simple" {
  firrtl.layer @A bind {
    firrtl.layer @B bind {
      firrtl.layer @C bind {}
    }
  }

  firrtl.module @Simple() {
    %a = firrtl.wire : !firrtl.uint<1>
    %b = firrtl.wire : !firrtl.uint<2>
    firrtl.layerblock @A {
      %aa = firrtl.node %a: !firrtl.uint<1>
      %c = firrtl.wire : !firrtl.uint<3>
      firrtl.layerblock @A::@B {
        %bb = firrtl.node %b: !firrtl.uint<2>
        %cc = firrtl.node %c: !firrtl.uint<3>
        firrtl.layerblock @A::@B::@C {
          %ccc = firrtl.node %cc: !firrtl.uint<3>
        }
      }
    }
  }
}

// CHECK-LABEL: firrtl.circuit "Simple"
//
// CHECK:      sv.verbatim "`include \22layers-Simple-A.sv\22\0A
// CHECK-SAME:   `include \22layers-Simple-A-B.sv\22\0A
// CHECK-SAME:   `ifndef layers_Simple_A_B_C\0A
// CHECK-SAME:   define layers_Simple_A_B_C"
// CHECK-SAME:   output_file = #hw.output_file<"layers-Simple-A-B-C.sv", excludeFromFileList>
// CHECK:      sv.verbatim "`include \22layers-Simple-A.sv\22\0A
// CHECK-SAME:   `ifndef layers_Simple_A_B\0A
// CHECK-SAME:   define layers_Simple_A_B"
// CHECK-SAME:   output_file = #hw.output_file<"layers-Simple-A-B.sv", excludeFromFileList>
// CHECK:      sv.verbatim "`ifndef layers_Simple_A\0A
// CHECK-SAME:   define layers_Simple_A"
// CHECK-SAME:   output_file = #hw.output_file<"layers-Simple-A.sv", excludeFromFileList>
//
// CHECK:      firrtl.module private @Simple_A_B_C(
// CHECK-NOT:  firrtl.module
// CHECK-SAME:   in %[[cc_port:[_a-zA-Z0-9]+]]: !firrtl.uint<3>
// CHECK-NEXT:   %ccc = firrtl.node %[[cc_port]]
// CHECK-NEXT: }
//
// CHECK:      firrtl.module private @Simple_A_B(
// CHECK-NOT:  firrtl.module
// CHECK-SAME:   in %[[b_port:[_a-zA-Z0-9]+]]: !firrtl.uint<2>
// CHECK-SAME:   in %[[c_port:[_a-zA-Z0-9]+]]: !firrtl.uint<3>
// CHECK-SAME:   out %[[cc_port:[_a-zA-Z0-9_]+]]: !firrtl.probe<uint<3>>
// CHECK-NEXT:   %bb = firrtl.node %[[b_port]]
// CHECK-NEXT:   %cc = firrtl.node %[[c_port]]
// CHECK-NEXT:   %0 = firrtl.ref.send %cc
// CHECK-NEXT:   firrtl.ref.define %[[cc_port]], %0
// CHECK-NEXT: }
//
// CHECK:      firrtl.module private @Simple_A(
// CHECK-NOT:  firrtl.module
// CHECK-SAME:   in %[[a_port:[_a-zA-Z0-9]+]]: !firrtl.uint<1>
// CHECK-SAME:   out %[[c_port:[_a-zA-Z0-9_]+]]: !firrtl.probe<uint<3>>
// CHECK-NEXT:   %aa = firrtl.node %[[a_port]]
// CHECK:        %[[c_ref:[_a-zA-Z0-9]+]] = firrtl.ref.send %c
// CHECK-NEXT:   firrtl.ref.define %[[c_port]], %[[c_ref]]
// CHECK-NEXT: }
//
// CHECK:      firrtl.module @Simple() {
// CHECK-NOT:  firrtl.module
// CHECK-NOT:    firrtl.layerblock
// CHECK:        %[[A_B_C_cc:[_a-zA-Z0-9_]+]] = firrtl.instance a_b_c {
// CHECK-SAME:     lowerToBind
// CHECK-SAME:     output_file = #hw.output_file<"layers-Simple-A-B-C.sv"
// CHECK-SAME:     excludeFromFileList
// CHECK-SAME:     @Simple_A_B_C(
// CHECK-NEXT:   %[[A_B_b:[_a-zA-Z0-9_]+]], %[[A_B_c:[_a-zA-Z0-9_]+]], %[[A_B_cc:[_a-zA-Z0-9_]+]] = firrtl.instance a_b {
// CHECK-SAME:     lowerToBind
// CHECK-SAME:     output_file = #hw.output_file<"layers-Simple-A-B.sv", excludeFromFileList>
// CHECK-SAME:     @Simple_A_B(
// CHECK-NEXT:   %[[A_B_cc_resolve:[_a-zA-Z0-9]+]] = firrtl.ref.resolve %[[A_B_cc]]
// CHECK-NEXT:   firrtl.matchingconnect %[[A_B_C_cc]], %[[A_B_cc_resolve]]
// CHECK-NEXT:   firrtl.matchingconnect %[[A_B_b]], %b
// CHECK-NEXT:   %[[A_a:[_a-zA-Z0-9_]+]], %[[A_c:[_a-zA-Z0-9_]+]] = firrtl.instance a {
// CHECK-SAME:     lowerToBind
// CHECK-SAME:     output_file = #hw.output_file<"layers-Simple-A.sv", excludeFromFileList>
// CHECK-SAME:     @Simple_A(
// CHECK-NEXT:   %[[A_c_resolve:[_a-zA-Z0-9]+]] = firrtl.ref.resolve %[[A_c]]
// CHECK-NEXT:   firrtl.matchingconnect %[[A_B_c]], %[[A_c_resolve]]
// CHECK-NEXT:   firrtl.matchingconnect %[[A_a]], %a
// CHECK:      }
//
// CHECK-DAG:  sv.verbatim "`endif // layers_Simple_A"
// CHECK-SAME:   output_file = #hw.output_file<"layers-Simple-A.sv", excludeFromFileList>
// CHECK-DAG:  sv.verbatim "`endif // layers_Simple_A_B"
// CHECK-SAME:   output_file = #hw.output_file<"layers-Simple-A-B.sv", excludeFromFileList>

// -----

firrtl.circuit "ModuleNameConflict" {
  firrtl.layer @A bind {}
  firrtl.module private @ModuleNameConflict_A() {}
  firrtl.module @ModuleNameConflict() {
    %a = firrtl.wire : !firrtl.uint<1>
    firrtl.instance foo @ModuleNameConflict_A()
    firrtl.layerblock @A {
      %b = firrtl.node %a : !firrtl.uint<1>
    }
  }
}

// CHECK-LABEL: firrtl.circuit "ModuleNameConflict"
//
// CHECK:       firrtl.module private @[[groupModule:[_a-zA-Z0-9_]+]](in
//
// CHECK:       firrtl.module @ModuleNameConflict()
// CHECK-NOT:   firrtl.module
// CHECK:         firrtl.instance foo @ModuleNameConflict_A()
// CHECK-NEXT:    firrtl.instance {{[_a-zA-Z0-9]+}} {lowerToBind,
// CHECK-SAME:      @[[groupModule]](

// -----
// Layerblock lowering must allow a value to be captured twice.
// https://github.com/llvm/circt/issues/6694

firrtl.circuit "CaptureHardwareMultipleTimes" {
  firrtl.layer @A bind {
    firrtl.layer @B bind {}
  }

  firrtl.extmodule @CaptureHardwareMultipleTimes ()

  // CHECK: firrtl.module private @[[A:.+]](in %[[p:.+]]: !firrtl.uint<1>)
  // CHECK:   %0 = firrtl.add %[[p]], %[[p]] : (!firrtl.uint<1>, !firrtl.uint<1>) -> !firrtl.uint<2>
  // CHECK: }
  // CHECK: firrtl.module @CaptureSrcTwice() {
  // CHECK:   %c0_ui1 = firrtl.constant 0 : !firrtl.uint<1>
  // CHECK:   %[[p:.+]] = firrtl.instance {{.+}} @[[A]]
  // CHECK:   firrtl.matchingconnect %[[p]], %c0_ui1 : !firrtl.uint<1>
  // CHECK: }
  firrtl.module @CaptureSrcTwice() {
    %c0_ui1 = firrtl.constant 0 : !firrtl.uint<1>
    firrtl.layerblock @A {
      %0 = firrtl.add %c0_ui1, %c0_ui1 : (!firrtl.uint<1>, !firrtl.uint<1>) -> !firrtl.uint<2>
    }
  }

  // CHECK: firrtl.module private @[[A:.+]](out %[[dst:.+]]: !firrtl.probe<uint<1>>, in %[[src:.+]]: !firrtl.uint<1>)
  // CHECK:   %0 = firrtl.ref.send %[[src]] : !firrtl.uint<1>
  // CHECK:   %w2 = firrtl.wire : !firrtl.probe<uint<1>>
  // CHECK:   firrtl.ref.define %[[dst]], %w2 : !firrtl.probe<uint<1>>
  // CHECK:   %1 = firrtl.ref.resolve %0 : !firrtl.probe<uint<1>>
  // CHECK: }
  // CHECK: firrtl.module @CaptureAsDstThenSrc() {
  // CHECK:   %w1 = firrtl.wire : !firrtl.probe<uint<1>>
  // CHECK:   %[[out:.+]], %[[in:.+]] = firrtl.instance {{.+}} @[[A]](out {{.+}}: !firrtl.probe<uint<1>>, in {{.+}}: !firrtl.uint<1>)
  // CHECK:   %0 = firrtl.ref.resolve %w1 : !firrtl.probe<uint<1>>
  // CHECK:   firrtl.matchingconnect %[[in]], %0 : !firrtl.uint<1>
  // CHECK:   firrtl.ref.define %w1, %[[out]] : !firrtl.probe<uint<1>>
  // CHECK: }
  firrtl.module @CaptureAsDstThenSrc() {
    %w1 = firrtl.wire : !firrtl.probe<uint<1>, @A>
    firrtl.layerblock @A {
      // capture first as a sink.
      %w2 = firrtl.wire : !firrtl.probe<uint<1>, @A>
      firrtl.ref.define %w1, %w2 : !firrtl.probe<uint<1>, @A>
      // Capture again, as a source.
      %2 = firrtl.ref.resolve %w1 : !firrtl.probe<uint<1>, @A>
    }
  }

  // CHECK: firrtl.module private @[[A:.+]](in %[[src:.+]]: !firrtl.uint<1>, out %[[dst:.+]]: !firrtl.probe<uint<1>>)
  // CHECK:   %0 = firrtl.ref.send %[[src]] : !firrtl.uint<1>
  // CHECK:   %1 = firrtl.ref.resolve %0 : !firrtl.probe<uint<1>>
  // CHECK:   %w2 = firrtl.wire : !firrtl.probe<uint<1>>
  // CHECK:   firrtl.ref.define %[[dst]], %w2 : !firrtl.probe<uint<1>>
  // CHECK: }
  // CHECK: firrtl.module @CaptureAsSrcThenDst() {
  // CHECK:   %w1 = firrtl.wire : !firrtl.probe<uint<1>>
  // CHECK:   %[[in:.+]], %[[out:.+]] = firrtl.instance {{.+}} @[[A]]
  // CHECK:   firrtl.ref.define %w1, %[[out]] : !firrtl.probe<uint<1>>
  // CHECK:   %0 = firrtl.ref.resolve %w1 : !firrtl.probe<uint<1>>
  // CHECK:   firrtl.matchingconnect %[[in]], %0 : !firrtl.uint<1>
  // CHECK: }
  firrtl.module @CaptureAsSrcThenDst() {
    %w1 = firrtl.wire : !firrtl.probe<uint<1>, @A>
    firrtl.layerblock @A {
      // capture first as a source.
      %2 = firrtl.ref.resolve %w1 : !firrtl.probe<uint<1>, @A>
      // capture again, as a sink.
      %w2 = firrtl.wire : !firrtl.probe<uint<1>, @A>
      firrtl.ref.define %w1, %w2 : !firrtl.probe<uint<1>, @A>
    }
  }
}

// -----
// HierPathOps are rewritten when operations with inner symbols inside
// layerblocks are moved into new modules.
// https://github.com/llvm/circt/issues/6717

firrtl.circuit "HierPathOps" {
  firrtl.extmodule @HierPathOps ()
  hw.hierpath @nla1 [@Foo::@foo_A]
  hw.hierpath @nla2 [@Foo::@bar, @Bar]
  hw.hierpath private @nla3 [@Foo::@baz]
  firrtl.layer @A  bind {}
  firrtl.module @Bar() {}
  firrtl.module @Foo() {
    %0 = firrtl.wire sym @foo_A : !firrtl.uint<1>
    firrtl.layerblock @A {
      firrtl.instance bar sym @bar @Bar()
      %1 = firrtl.wire sym @baz : !firrtl.uint<1>
    }
  }
}

// CHECK-LABEL: firrtl.circuit "HierPathOps"
//
// CHECK:         hw.hierpath         @nla1 [@Foo::@foo_A]
// CHECK-NEXT:    hw.hierpath         @nla2 [@Foo::@[[inst_sym:[_A-Za-z0-9]+]], @[[mod_sym:[_A-Za-z0-9_]+]]::@bar, @Bar]
// CHECK-NEXT:    hw.hierpath private @nla3 [@Foo::@[[inst_sym]], @[[mod_sym]]::@baz]
//
// CHECK:         firrtl.module {{.*}} @[[mod_sym]]()
// CHECK-NEXT:      firrtl.instance bar sym @bar
// CHECK-NEXT:      firrtl.wire sym @baz
//
// CHECK:         firrtl.module @Foo()
// CHECK-NEXT:      firrtl.wire sym @foo_A :
// CHECK-NEXT:      firrtl.instance {{.*}} sym @[[inst_sym]]

// -----
// Check the output file behavior when both a DUT and a testbench directory are
// specified.  In the test below, Foo is the testbench and Bar is the DUT.

firrtl.circuit "Foo" attributes {
  annotations = [
    {
      class = "sifive.enterprise.firrtl.TestBenchDirAnnotation",
      dirname = "testbench"
    }
  ]
} {
  firrtl.layer @A  bind attributes {output_file = #hw.output_file<"testbench/", excludeFromFileList>} {}
  firrtl.module @Bar() attributes {
    annotations = [
      {
        class = "sifive.enterprise.firrtl.MarkDUTAnnotation"
      }
    ]
  } {
    firrtl.layerblock @A {
      %a = firrtl.wire : !firrtl.uint<1>
    }
  }
  firrtl.module @Foo() {
    firrtl.layerblock @A {
      %a = firrtl.wire : !firrtl.uint<1>
    }
  }
}

// CHECK-LABEL: firrtl.circuit "Foo"
//
// CHECK:       sv.verbatim
// CHECK-SAME:    #hw.output_file<"testbench{{/|\\\\}}layers-Foo-A.sv", excludeFromFileList>
//
// CHECK:       firrtl.module {{.*}} @Bar_A
// CHECK-SAME:    #hw.output_file<"testbench{{/|\\\\}}", excludeFromFileList>
// CHECK:       firrtl.module {{.*}} @Foo_A
// CHECK-SAME:    #hw.output_file<"testbench{{/|\\\\}}", excludeFromFileList>
//
// CHECK:       sv.verbatim
// CHECK-SAME:    #hw.output_file<"testbench{{/|\\\\}}layers-Foo-A.sv", excludeFromFileList>


// -----
// Check that we correctly implement the verilog header and footer for B.

firrtl.circuit "Foo" {
  firrtl.layer @A bind {
    firrtl.layer @X bind {}
  }
  firrtl.layer @B bind {}
  firrtl.module @Foo() {}
}

// CHECK: firrtl.circuit "Foo" {
// CHECK:   sv.verbatim "`ifndef layers_Foo_B\0A`define layers_Foo_B" {output_file = #hw.output_file<"layers-Foo-B.sv", excludeFromFileList>}
// CHECK:   firrtl.module @Foo() {
// CHECK:   }
// CHECK:   sv.verbatim "`endif // layers_Foo_B" {output_file = #hw.output_file<"layers-Foo-B.sv", excludeFromFileList>}
// CHECK: }

// -----

// Check rwprobe ops are updated.
// CHECK-LABEL: circuit "RWTH"
firrtl.circuit "RWTH" {
  firrtl.layer @T  bind { }
  firrtl.module @RWTH() attributes {convention = #firrtl<convention scalarized>, layers = [@T]} {
    %d_p = firrtl.instance d @DUT(out p: !firrtl.rwprobe<uint<1>, @T>)
    %one = firrtl.constant 1 : !firrtl.uint<1>
    firrtl.ref.force_initial %one, %d_p, %one: !firrtl.uint<1>, !firrtl.rwprobe<uint<1>, @T>, !firrtl.uint<1>
  }
//      CHECK:    firrtl.module private @DUT_T(out %p: !firrtl.rwprobe<uint<1>>) {
// CHECK-NEXT:      %w = firrtl.wire sym @[[SYM:.+]] : !firrtl.uint<1>
// CHECK-NEXT:      %0 = firrtl.ref.rwprobe <@DUT_T::@[[SYM]]> : !firrtl.rwprobe<uint<1>>
// CHECK-NEXT:      firrtl.ref.define %p, %0 : !firrtl.rwprobe<uint<1>>
// CHECK-NEXT:    }
// CHECK-NEXT:    firrtl.module @DUT(out %p: !firrtl.rwprobe<uint<1>>) attributes {convention = #firrtl<convention scalarized>} {
// CHECK-NEXT:      %t_p = firrtl.instance t sym @t {lowerToBind, output_file = #hw.output_file<"layers-RWTH-T.sv", excludeFromFileList>} @DUT_T(out p: !firrtl.rwprobe<uint<1>>)
// CHECK-NEXT:      firrtl.ref.define %p, %t_p : !firrtl.rwprobe<uint<1>>
// CHECK-NEXT:    }

  firrtl.module @DUT(out %p: !firrtl.rwprobe<uint<1>, @T>) attributes {convention = #firrtl<convention scalarized>} {
    firrtl.layerblock @T {
      %w = firrtl.wire sym @sym : !firrtl.uint<1>
      %0 = firrtl.ref.rwprobe <@DUT::@sym> : !firrtl.rwprobe<uint<1>>
      %1 = firrtl.ref.cast %0 : (!firrtl.rwprobe<uint<1>>) -> !firrtl.rwprobe<uint<1>, @T>
      firrtl.ref.define %p, %1 : !firrtl.rwprobe<uint<1>, @T>
    }
  }
}
