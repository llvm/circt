// RUN: circt-opt -firrtl-lower-layers -split-input-file %s | FileCheck %s

// CHECK-LABEL: firrtl.circuit "Test"
firrtl.circuit "Test" {
  firrtl.module @Test() {}

  firrtl.layer @A bind {
    firrtl.layer @B bind {
      firrtl.layer @C bind {}
    }
  }
  firrtl.layer @B bind {}

  firrtl.domain @ClockDomain

  firrtl.extmodule @Foo(out o : !firrtl.probe<uint<1>, @A>) attributes {knownLayers=[@A]}

  //===--------------------------------------------------------------------===//
  // Removal of Probe Colors
  //===--------------------------------------------------------------------===//

  // CHECK-LABEL: @ColoredPorts(out %o: !firrtl.probe<uint<1>>)
  firrtl.module @ColoredPorts(out %o: !firrtl.probe<uint<1>, @A>) {}

  // CHECK-LABEL: @ExtColoredPorts(out o: !firrtl.probe<uint<1>>)
  firrtl.extmodule @ExtColoredPorts(out o: !firrtl.probe<uint<1>, @A>) attributes {knownLayers=[@A]}

  // CHECK-LABEL: @ColoredPortsOnInstances
  firrtl.module @ColoredPortsOnInstances() {
    // CHECK: %foo_o = firrtl.instance foo @ColoredPorts(out o: !firrtl.probe<uint<1>>)
   %foo_o = firrtl.instance foo @ColoredPorts(out o: !firrtl.probe<uint<1>, @A>)
  }

  // CHECK-LABEL: @ColoredThings
  firrtl.module @ColoredThings() {
    // CHECK: %0 = firrtl.wire : !firrtl.probe<bundle<f: uint<1>>>
    %0 = firrtl.wire : !firrtl.probe<bundle<f: uint<1>>, @A>
  }

    // CHECK-LABEL: @ColoredThingUnderWhen
  firrtl.module @ColoredThingUnderWhen(in %b : !firrtl.uint<1>) {
    // CHECK: firrtl.when %b : !firrtl.uint<1>
    firrtl.when %b : !firrtl.uint<1> {
      // CHECK: %0 = firrtl.wire : !firrtl.probe<bundle<f: uint<1>>>
      %0 = firrtl.wire : !firrtl.probe<bundle<f: uint<1>>, @A>
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

  // CHECK-LABEL firrtl.module @WithLayerBlock
  firrtl.module @WithLayerBlock() {
    // CHECK-NOT firrtl.layerblock @GoodbyeCruelWorld
    firrtl.layerblock @GoodbyeCruelWorld {
    }
  }

  //===--------------------------------------------------------------------===//
  // Capture
  //===--------------------------------------------------------------------===//

  // CHECK:      hw.hierpath private @[[CaptureHardware_c0_ui1_path:.+]] [@CaptureHardware::@[[CaptureHardware_c0_ui1_sym:.+]]]
  // CHECK-NEXT: hw.hierpath private @[[CaptureHardware_c1_ui1_path:.+]] [@CaptureHardware::@[[CaptureHardware_c1_ui1_sym:.+]]]
  // CHECK-NEXT: firrtl.module private @CaptureHardware_A() {
  // CHECK-NEXT:   %[[c1_ui1:.+]] = firrtl.xmr.deref @[[CaptureHardware_c1_ui1_path]] : !firrtl.uint<1>
  // CHECK-NEXT:   %[[c0_ui1:.+]] = firrtl.xmr.deref @[[CaptureHardware_c0_ui1_path]] : !firrtl.uint<1>
  // CHECK-NEXT:   firrtl.add %[[c0_ui1]], %[[c1_ui1]] : (!firrtl.uint<1>, !firrtl.uint<1>) -> !firrtl.uint<2>
  // CHECK-NEXT: }
  // CHECK-NEXT: firrtl.module @CaptureHardware() {
  // CHECK-NEXT:   %c0_ui1 = firrtl.constant 0 : !firrtl.uint<1>
  // CHECK-NEXT:   %[[c0_ui1_node:.+]] = firrtl.node sym @[[CaptureHardware_c0_ui1_sym]] %c0_ui1
  // CHECK-NEXT:   %c1_ui1 = firrtl.constant 1 : !firrtl.uint<1>
  // CHECK-NEXT:   %[[c1_ui1_node:.+]] = firrtl.node sym @[[CaptureHardware_c1_ui1_sym]] %c1_ui1
  // CHECK-NEXT:   firrtl.instance {{.+}} {doNotPrint, output_file = #hw.output_file<"layers-CaptureHardware-A.sv", excludeFromFileList>} @CaptureHardware_A()
  // CHECK-NEXT: }
  firrtl.module @CaptureHardware() {
    %c0_ui1 = firrtl.constant 0 : !firrtl.uint<1>
    %c1_ui1 = firrtl.constant 1 : !firrtl.uint<1>
    firrtl.layerblock @A {
      %0 = firrtl.add %c0_ui1, %c1_ui1 : (!firrtl.uint<1>, !firrtl.uint<1>) -> !firrtl.uint<2>
    }
  }

  // CHECK: hw.hierpath private @[[CapturePort_port_path:.+]] [@CapturePort::@[[CapturePort_port_sym:.+]]]
  // CHECK: firrtl.module private @CapturePort_A() {
  // CHECK:   %[[port:.+]] = firrtl.xmr.deref @[[CapturePort_port_path]] : !firrtl.uint<1>
  // CHECK:   %x = firrtl.node %[[port]] : !firrtl.uint<1>
  // CHECK: }
  // CHECK: firrtl.module @CapturePort(in %in: !firrtl.uint<1> sym @[[CapturePort_port_sym]]) {
  // CHECK:   firrtl.instance {{.+}} {doNotPrint, output_file = #hw.output_file<"layers-CapturePort-A.sv", excludeFromFileList>} @CapturePort_A
  // CHECK: }
  firrtl.module @CapturePort(in %in: !firrtl.uint<1>){
    firrtl.layerblock @A {
      %x = firrtl.node %in : !firrtl.uint<1>
    }
  }

  // CHECK:      hw.hierpath private @[[CaptureConnect_a_path:.+]] [@CaptureConnect::@[[CaptureConnect_a_sym:.+]]]
  // CHECK-NEXT: firrtl.module private @CaptureConnect_A()
  // CHECK-NEXT:   %[[a:.+]] = firrtl.xmr.deref @[[CaptureConnect_a_path]] : !firrtl.uint<1>
  // CHECK-NEXT:   %b = firrtl.wire : !firrtl.uint<1>
  // CHECK-NEXT:   firrtl.connect %b, %[[a]] : !firrtl.uint<1>
  // CHECK-NEXT: }
  // CHECK-NEXT: firrtl.module @CaptureConnect() {
  // CHECK-NEXT:   %a = firrtl.wire : !firrtl.uint<1>
  // CHECK-NEXT:   %a_0 = firrtl.node sym @[[CaptureConnect_a_sym]] %a {name = "a"} : !firrtl.uint<1>
  // CHECK-NEXT:   firrtl.instance {{.+}} sym @{{.+}} {doNotPrint, output_file = #hw.output_file<"layers-CaptureConnect-A.sv", excludeFromFileList>} @CaptureConnect_A()
  // CHECK-NEXT: }
  firrtl.module @CaptureConnect() {
    %a = firrtl.wire : !firrtl.uint<1>
    firrtl.layerblock @A {
      %b = firrtl.wire : !firrtl.uint<1>
      firrtl.connect %b, %a : !firrtl.uint<1>, !firrtl.uint<1>
    }
  }

  // CHECK:      firrtl.module private @CaptureProbeSrc_A()
  // CHECK-NEXT:   %0 = firrtl.xmr.deref @xmrPath : !firrtl.uint<1>
  // CHECK-NEXT: }
  // CHECK-NEXT: firrtl.module @CaptureProbeSrc() {
  // CHECK-NEXT:   %w = firrtl.wire : !firrtl.uint<1>
  // CHECK-NEXT:   %w_probe = firrtl.node sym @sym interesting_name %w : !firrtl.uint<1>
  // CHECK-NEXT:   firrtl.instance {{.+}} {doNotPrint, output_file = #hw.output_file<"layers-CaptureProbeSrc-A.sv", excludeFromFileList>} @CaptureProbeSrc_A
  // CHECK-NEXT: }
  hw.hierpath private @xmrPath [@CaptureProbeSrc::@sym]
  firrtl.module @CaptureProbeSrc() {
    %w = firrtl.wire : !firrtl.uint<1>
    %w_probe = firrtl.node sym @sym interesting_name %w : !firrtl.uint<1>
    firrtl.layerblock @A {
      %0 = firrtl.xmr.deref @xmrPath : !firrtl.uint<1>
    }
  }

  // CHECK:      hw.hierpath private @[[NestedCapture_x_path:.+]] [@NestedCapture::@[[inst:.+]], @NestedCapture_A::@[[NestedCapture_a_sym:.+]]]
  // CHECK-NEXT: firrtl.module private @NestedCapture_A_B()
  // CHECK-NEXT:   %0 = firrtl.xmr.deref @[[NestedCapture_x_path]]
  // CHECK-NEXT:   %1 = firrtl.node %0 : !firrtl.uint<1>
  // CHECK-NEXT: }
  // CHECK-NEXT: firrtl.module private @NestedCapture_A() {
  // CHECK-NEXT:   %x = firrtl.wire : !firrtl.uint<1>
  // CHECK-NEXT:   %x_0 = firrtl.node sym @[[NestedCapture_a_sym]] %x {name = "x"}
  // CHECK-NEXT: }
  // CHECK-NEXT: firrtl.module @NestedCapture() {
  // CHECK-NEXT:   firrtl.instance {{.+}} {doNotPrint, output_file = #hw.output_file<"layers-NestedCapture-A-B.sv", excludeFromFileList>} @NestedCapture_A_B()
  // CHECK-NEXT:   firrtl.instance {{.+}} sym @[[inst]] {doNotPrint, output_file = #hw.output_file<"layers-NestedCapture-A.sv", excludeFromFileList>} @NestedCapture_A()
  // CHECK-NEXT: }
  firrtl.module @NestedCapture() {
    firrtl.layerblock @A {
      %x = firrtl.wire : !firrtl.uint<1>
      firrtl.layerblock @A::@B {
        %0 = firrtl.node %x : !firrtl.uint<1>
      }
    }
  }

  // CHECK:      hw.hierpath private @[[WhenUnderLayer_x_path:.+]] [@WhenUnderLayer::@[[WhenUnderLayer_x_sym:.+]]]
  // CHECK-NEXT: firrtl.module private @WhenUnderLayer_A()
  // CHECK-NEXT:   %0 = firrtl.xmr.deref @[[WhenUnderLayer_x_path]] : !firrtl.uint<1>
  // CHECK-NEXT:   %c1_ui1 = firrtl.constant 1 : !firrtl.uint<1>
  // CHECK-NEXT:   firrtl.when %0 : !firrtl.uint<1> {
  // CHECK-NEXT:     %1 = firrtl.add %0, %c1_ui1 : (!firrtl.uint<1>, !firrtl.uint<1>) -> !firrtl.uint<2>
  // CHECK-NEXT:   }
  // CHECK-NEXT: }
  // CHECK-NEXT: firrtl.module @WhenUnderLayer() {
  // CHECK-NEXT:   %x = firrtl.wire : !firrtl.uint<1>
  // CHECK-NEXT:   %x_0 = firrtl.node sym @[[WhenUnderLayer_x_sym]] %x {name = "x"}
  // CHECK-NEXT:   firrtl.instance {{.+}} sym @{{.+}} {doNotPrint, output_file = #hw.output_file<"layers-WhenUnderLayer-A.sv", excludeFromFileList>} @WhenUnderLayer_A
  // CHECK-NEXT: }
  firrtl.module @WhenUnderLayer() {
    %x = firrtl.wire : !firrtl.uint<1>
    firrtl.layerblock @A {
      %c1_ui1 = firrtl.constant 1 : !firrtl.uint<1>
      firrtl.when %x : !firrtl.uint<1> {
        %0 = firrtl.add %x, %c1_ui1 : (!firrtl.uint<1>, !firrtl.uint<1>) -> !firrtl.uint<2>
      }
    }
  }

  // Test that subfield, subindex, and subaccess are moved out of layerblocks to
  // avoid capturing non-passive types.
  //
  // CHECK:      hw.hierpath private @[[SubOpsInLayerBlock_sub_path:.+]] [@SubOpsInLayerBlock::@[[SubOpsInLayerBlock_sub_sym:.+]]]
  // CHECK-NEXT: firrtl.module private @SubOpsInLayerBlock_A() {
  // CHECK-NEXT:   %0 = firrtl.xmr.deref @[[SubOpsInLayerBlock_sub_path]]
  // CHECK-NEXT:   firrtl.node %0
  // CHECK-NEXT: }
  // CHECK:      firrtl.module @SubOpsInLayerBlock(
  // CHECK-NEXT:   %0 = firrtl.subaccess %a[%b]
  // CHECK-NEXT:   %1 = firrtl.subindex %0[0]
  // CHECK-NEXT:   %2 = firrtl.subfield %1[a]
  // CHECK-NEXT:   %3 = firrtl.node %2 :
  // CHECK-NEXT:   %_layer_probe = firrtl.node sym @[[SubOpsInLayerBlock_sub_sym]] %3
  firrtl.module @SubOpsInLayerBlock(
    in %a: !firrtl.vector<vector<bundle<a: uint<1>, b flip: uint<2>>, 2>, 2>,
    in %b: !firrtl.uint<1>
  ) {
    firrtl.layerblock @A {
      %0 = firrtl.subaccess %a[%b] : !firrtl.vector<vector<bundle<a: uint<1>, b flip: uint<2>>, 2>, 2>, !firrtl.uint<1>
      %1 = firrtl.subindex %0[0] : !firrtl.vector<bundle<a: uint<1>, b flip: uint<2>>, 2>
      %2 = firrtl.subfield %1[a] : !firrtl.bundle<a: uint<1>, b flip: uint<2>>
      %3 = firrtl.node %2 : !firrtl.uint<1>
      %4 = firrtl.node %3 : !firrtl.uint<1>
    }
  }

  // Test that subfield and subindex ops are captured as normal when the input is passive.
  //
  // CHECK:      hw.hierpath private @[[vec_path:.+]] [@SubOpsCapturingPassiveInput::@[[vec_sym:.+]]]
  // CHECK-NEXT: hw.hierpath private @[[bun_path:.+]] [@SubOpsCapturingPassiveInput::@[[bun_sym:.+]]]
  // CHECK-NEXT: firrtl.module private @SubOpsCapturingPassiveInput_A() {
  // CHECK-NEXT:   %0 = firrtl.xmr.deref @[[bun_path]] : !firrtl.bundle<a: uint<1>>
  // CHECK-NEXT:   %1 = firrtl.xmr.deref @[[vec_path]] : !firrtl.vector<uint<1>, 1>
  // CHECK-NEXT:   %2 = firrtl.subindex %1[0] : !firrtl.vector<uint<1>, 1>
  // CHECK-NEXT:   %3 = firrtl.subfield %0[a] : !firrtl.bundle<a: uint<1>>
  // CHECK-NEXT: }
  // CHECK:      firrtl.module @SubOpsCapturingPassiveInput(
  // CHECK-SAME:     in %vec: !firrtl.vector<uint<1>, 1> sym @[[vec_sym]],
  // CHECK-SAME:     in %bun: !firrtl.bundle<a: uint<1>> sym @[[bun_sym]]) {
  // CHECK-NEXT:   firrtl.instance a {{.*}}
  // CHECK-NEXT: }
  firrtl.module @SubOpsCapturingPassiveInput (
    in %vec: !firrtl.vector<uint<1>, 1>,
    in %bun: !firrtl.bundle<a: uint<1>>
  ) {
    firrtl.layerblock @A {
      %0 = firrtl.subindex %vec[0] : !firrtl.vector<uint<1>, 1>
      %1 = firrtl.subfield %bun[a] : !firrtl.bundle<a: uint<1>>
    }
  }

  // Test that subaccess op operands are captured as normal when the input is passive.
  //
  // CHECK:      hw.hierpath private @[[input_path:.+]] [@SubaccessOpCapturingPassiveInput::@[[input_sym:.+]]]
  // CHECK-NEXT: hw.hierpath private @[[index_path:.+]] [@SubaccessOpCapturingPassiveInput::@[[index_sym:.+]]]
  // CHECK-NEXT: firrtl.module private @SubaccessOpCapturingPassiveInput_A() {
  // CHECK-NEXT:  %0 = firrtl.xmr.deref @[[index_path]] : !firrtl.uint<1>
  // CHECK-NEXT:  %1 = firrtl.xmr.deref @[[input_path]] : !firrtl.vector<uint<1>, 2>
  // CHECK-NEXT:  %2 = firrtl.subaccess %1[%0] : !firrtl.vector<uint<1>, 2>, !firrtl.uint<1>
  // CHECK-NEXT: }
  // CHECK:      firrtl.module @SubaccessOpCapturingPassiveInput(
  // CHECK-SAME:    in %input: !firrtl.vector<uint<1>, 2> sym @[[input_sym]],
  // CHECK-SAME:    in %index: !firrtl.uint<1> sym @[[index_sym]]) {
  // CHECK-NEXT:  firrtl.instance a {{.*}}
  // CHECK-NEXT: }
  firrtl.module @SubaccessOpCapturingPassiveInput (
    in %input: !firrtl.vector<uint<1>, 2>,
    in %index: !firrtl.uint<1>
  ) {
    firrtl.layerblock @A {
      %0 = firrtl.subaccess %input[%index] : !firrtl.vector<uint<1>, 2>, !firrtl.uint<1>
    }
  }

  // CHECK:      hw.hierpath private @[[CaptureWhen2_a_path:.+]] [@CaptureWhen2::@[[CaptureWhen2_a_sym:.+]]]
  // CHECK-NEXT: hw.hierpath private @[[CaptureWhen2_cond_path:.+]] [@CaptureWhen2::@[[CaptureWhen2_cond_sym:.+]]]
  // CHECK-NEXT: firrtl.module private @CaptureWhen2_A() {
  // CHECK-NEXT:   %0 = firrtl.xmr.deref @[[CaptureWhen2_cond_path]]
  // CHECK-NEXT:   %1 = firrtl.xmr.deref @[[CaptureWhen2_a_path]]
  // CHECK-NEXT:   firrtl.when %0 {{.*}} {
  // CHECK-NEXT:     %b = firrtl.node %1

  // CHECK:      firrtl.module @CaptureWhen2(
  // CHECK-NEXT:   %a = firrtl.wire
  // CHECK-NEXT:   %a_0 = firrtl.node {{.*}} %a
  // CHECK-NEXT:   firrtl.instance a
  firrtl.module @CaptureWhen2(in %cond: !firrtl.uint<1>) {
    %a = firrtl.wire : !firrtl.uint<1>
    firrtl.layerblock @A {
      firrtl.when %cond : !firrtl.uint<1> {
        %b = firrtl.node %a : !firrtl.uint<1>
      }
    }
  }

  // Capture of a zero-width value creates a local zero-width constant zero.
  //
  // CHECK:      firrtl.module private @ZeroWidthCapture_A() {
  // CHECK-NEXT:   %c0_ui0 = firrtl.constant 0 : !firrtl.uint<0>
  // CHECK-NEXT:   %b = firrtl.node %c0_ui0 : !firrtl.uint<0>
  // CHECK-NEXT: }
  // CHECK:      firrtl.module @ZeroWidthCapture() {
  // CHECK-NEXT:   %a = firrtl.wire : !firrtl.uint<0>
  // CHECK-NEXT:   firrtl.instance a
  firrtl.module @ZeroWidthCapture() {
    %a = firrtl.wire : !firrtl.uint<0>
    firrtl.layerblock @A {
      %b = firrtl.node %a : !firrtl.uint<0>
    }
  }

  // Port capture needs to create a node.
  //
  // CHECK:      hw.hierpath private @[[InstancePortCapture_ext_a_path:.+]] [@InstancePortCapture::@[[InstancePortCapture_ext_a_sym:.+]]]
  // CHECK-NEXT: hw.hierpath private @[[InstancePortCapture_ext_b_path:.+]] [@InstancePortCapture::@[[InstancePortCapture_ext_b_sym:.+]]]
  // CHECK-NEXT: firrtl.module private @InstancePortCapture_A() {
  // CHECK-NEXT:   %0 = firrtl.xmr.deref @[[InstancePortCapture_ext_b_path]]
  // CHECK-NEXT:   %1 = firrtl.xmr.deref @[[InstancePortCapture_ext_a_path]]
  // CHECK-NEXT:   %a = firrtl.node %1
  // CHECK-NEXT:   %b = firrtl.node %0
  // CHECK-NEXT: }
  //
  // CHECK:      firrtl.module @InstancePortCapture() {
  // CHECK-NEXT:   %ext_a, %ext_b = firrtl.instance ext @InstancePortCapture_ext
  // CHECK-NEXT:   %ext_b_0 = firrtl.node sym @[[InstancePortCapture_ext_b_sym]] %ext_b {name = "ext_b"}
  // CHECK-NEXT:   %ext_a_1 = firrtl.node sym @[[InstancePortCapture_ext_a_sym]] %ext_a {name = "ext_a"}
  // CHECK-NEXT:   firrtl.instance {{.*}}
  // CHECK-NEXT: }
  firrtl.extmodule @InstancePortCapture_ext(
    in a: !firrtl.uint<1>,
    out b: !firrtl.uint<1>
  )
  firrtl.module @InstancePortCapture() {
    %ext_a, %ext_b = firrtl.instance ext @InstancePortCapture_ext(
      in a: !firrtl.uint<1>,
      out b: !firrtl.uint<1>
    )
    firrtl.layerblock @A {
      %a = firrtl.node %ext_a : !firrtl.uint<1>
      %b = firrtl.node %ext_b : !firrtl.uint<1>
    }
  }

  // Test that a single domain type capture works.
  //
  // CHECK:      firrtl.module private @DomainCapture_A(
  // CHECK-SAME:   in %A: !firrtl.domain of @ClockDomain
  //
  // CHECK-NEXT:   %a = firrtl.wire {domainKind = @ClockDomain} : !firrtl.domain
  // CHECK-NEXT:   firrtl.domain.define %a, %A
  //
  // CHECK:      firrtl.module @DomainCapture(
  // CHECK-SAME:   in %A: !firrtl.domain of @ClockDomain
  //
  // CHECK-NEXT:   %a_A = firrtl.instance a {{.*}} @DomainCapture_A(
  // CHECK-NEXT:   firrtl.domain.define %a_A, %A
  firrtl.module @DomainCapture(
    in %A: !firrtl.domain of @ClockDomain
  ) {
    firrtl.layerblock @A {
      %a = firrtl.wire {domainKind = @ClockDomain} : !firrtl.domain
      firrtl.domain.define %a, %A
    }
  }

  // Test that a layer capture from a nested layer works.  This is specifically
  // checking that an unnuecessary intermediary module/instance with squiggled
  // ports isn't created.
  //
  // CHECK:      firrtl.module private @DomainCaptureTwoLevels_A_B(
  // CHECK-SAME:   in %A: !firrtl.domain of @ClockDomain
  //
  // CHECK-NEXT:   %a = firrtl.wire {domainKind = @ClockDomain} : !firrtl.domain
  // CHECK-NEXT:   firrtl.domain.define %a, %A
  //
  // CHECK-NOT:  firrtl.module private @DomainCaptureTwoLevels_A(
  //
  // CHECK:      firrtl.module @DomainCaptureTwoLevels(
  // CHECK-SAME:   in %A: !firrtl.domain of @ClockDomain
  //
  // CHECK-NEXT:   %a_b_A = firrtl.instance a_b {{.*}} @DomainCaptureTwoLevels_A_B(
  // CHECK-NEXT:   firrtl.domain.define %a_b_A, %A
  firrtl.module @DomainCaptureTwoLevels(
    in %A: !firrtl.domain of @ClockDomain
  ) {
    firrtl.layerblock @A {
      firrtl.layerblock @A::@B {
        %a = firrtl.wire {domainKind = @ClockDomain} : !firrtl.domain
        firrtl.domain.define %a, %A
      }
    }
  }

  // CHECK:      firrtl.module private @DomainCaptureChildCapturesParent_A_B(
  // CHECK-SAME:   in %anonDomain: !firrtl.domain of @ClockDomain
  //
  // CHECK-NEXT:   %a = firrtl.wire {domainKind = @ClockDomain} : !firrtl.domain
  // CHECK-NEXT:   firrtl.domain.define %a, %anonDomain
  //
  // CHECK:      firrtl.module private @DomainCaptureChildCapturesParent_A(
  // CHECK-SAME:   out %anonDomain: !firrtl.domain of @ClockDomain
  //
  // CHECK-NEXT:   %0 = firrtl.domain.anon
  // CHECK-NEXT:   firrtl.domain.define %anonDomain, %0
  //
  // CHECK:      firrtl.module @DomainCaptureChildCapturesParent(
  //
  // CHECK-NEXT:   %a_b_anonDomain = firrtl.instance a_b {{.*}} @DomainCaptureChildCapturesParent_A_B
  // CHECK-NEXT:   %a_anonDomain = firrtl.instance a {{.*}} @DomainCaptureChildCapturesParent_A
  // CHECK-NEXT:   firrtl.domain.define %a_b_anonDomain, %a_anonDomain
  firrtl.module @DomainCaptureChildCapturesParent(
    in %A: !firrtl.domain of @ClockDomain
  ) {
    firrtl.layerblock @A {
      %0 = firrtl.domain.anon : !firrtl.domain of @ClockDomain
      firrtl.layerblock @A::@B {
        %a = firrtl.wire {domainKind = @ClockDomain} : !firrtl.domain
        firrtl.domain.define %a, %0
      }
    }
  }

  // CHECK:      firrtl.module private @DomainCaptureInterleaved_A_B(
  // CHECK-SAME:   in %b: !firrtl.domain of @ClockDomain
  //
  // CHECK-NEXT:   %a = firrtl.wire {domainKind = @ClockDomain} : !firrtl.domain
  // CHECK-NEXT:   firrtl.domain.define %a, %b
  //
  // CHECK:      firrtl.module private @DomainCaptureInterleaved_A(
  // CHECK-SAME:   in %A: !firrtl.domain of @ClockDomain
  // CHECK-SAME:   out %b: !firrtl.domain of @ClockDomain
  //
  // CHECK-NEXT:   %b_0 = firrtl.wire {domainKind = @ClockDomain, name = "b"} : !firrtl.domain
  // CHECK-NEXT:   firrtl.domain.define %b_0, %A
  // CHECK-NEXT:   firrtl.domain.define %b, %b_0
  //
  // CHECK:      firrtl.module @DomainCaptureInterleaved(
  //
  // CHECK-NEXT:   %a_b_b = firrtl.instance a_b {{.*}} @DomainCaptureInterleaved_A_B
  // CHECK-NEXT:   %a_A, %a_b = firrtl.instance a {{.*}} @DomainCaptureInterleaved_A
  // CHECK-NEXT:   firrtl.domain.define %a_A, %A
  // CHECK-NEXT:   firrtl.domain.define %a_b_b, %a_b
  firrtl.module @DomainCaptureInterleaved(
    in %A: !firrtl.domain of @ClockDomain
  ) {
    firrtl.layerblock @A {
      %b = firrtl.wire {domainKind = @ClockDomain} : !firrtl.domain
      firrtl.domain.define %b, %A
      firrtl.layerblock @A::@B {
        %a = firrtl.wire {domainKind = @ClockDomain} : !firrtl.domain
        firrtl.domain.define %a, %b
      }
    }
  }

  // CHECK:      firrtl.module private @DomainCaptureInterleaved2_A_B(
  // CHECK-SAME:   in %A: !firrtl.domain of @ClockDomain
  //
  // CHECK-NEXT:   %wire = firrtl.wire {domainKind = @ClockDomain} : !firrtl.domain
  // CHECK-NEXT:   firrtl.domain.define %wire, %A
  //
  // CHECK:      firrtl.module private @DomainCaptureInterleaved2_A(
  // CHECK-SAME:   in %A: !firrtl.domain of @ClockDomain
  //
  // CHECK-NEXT:   %b = firrtl.wire {domainKind = @ClockDomain} : !firrtl.domain
  // CHECK-NEXT:   firrtl.domain.define %b, %A
  //
  // CHECK:      firrtl.module @DomainCaptureInterleaved2(
  //
  // CHECK-NEXT:   %a_b_A = firrtl.instance a_b {{.*}} @DomainCaptureInterleaved2_A_B
  // CHECK-NEXT:   firrtl.domain.define %a_b_A, %A
  // CHECK-NEXT:   %a_A = firrtl.instance a {{.*}} @DomainCaptureInterleaved2_A
  // CHECK-NEXT:   firrtl.domain.define %a_A, %A
  firrtl.module @DomainCaptureInterleaved2(
    in %A: !firrtl.domain of @ClockDomain
  ) {
    firrtl.layerblock @A {
      %b = firrtl.wire {domainKind = @ClockDomain} : !firrtl.domain
      firrtl.domain.define %b, %A
      firrtl.layerblock @A::@B {
        %wire = firrtl.wire {domainKind = @ClockDomain} : !firrtl.domain
        firrtl.domain.define %wire, %A
      }
    }
  }

  //===--------------------------------------------------------------------===//
  // Cloning of special operations
  //===--------------------------------------------------------------------===//

  // An FString operation is outside the layer block.  This needs to be cloned.
  //
  // CHECK: firrtl.module private @[[A:.+]]() {
  // CHECK:   %time = firrtl.fstring.time
  // CHECK:   firrtl.printf %clock, %c1_ui1, "{{.*}}" (%time)
  // CHECK: }
  // CHECK: firrtl.module @FStringOp
  firrtl.module @FStringOp() {
    %time = firrtl.fstring.time : !firrtl.fstring
    firrtl.layerblock @A {
       %clock = firrtl.wire : !firrtl.clock
      %c1_ui1 = firrtl.constant 1 : !firrtl.uint<1>
      firrtl.printf %clock, %c1_ui1, "{{}}" (%time) : !firrtl.clock, !firrtl.uint<1>, !firrtl.fstring
    }
  }

  // XMR Ref ops used by force_initial are cloned.
  //
  // CHECK:      firrtl.module private @XmrRef_A()
  // CHECK-NEXT:   %0 = firrtl.xmr.ref @XmrRef_path : !firrtl.rwprobe<uint<1>>
  // CHECK-NEXT:   %a = firrtl.wire
  // CHECK-NEXT:   %c1_ui1 = firrtl.constant 1
  // CHECK-NEXT:   firrtl.ref.force_initial %c1_ui1, %0, %c1_ui1
  hw.hierpath private @XmrRef_path [@XmrRef::@a]
  firrtl.module @XmrRef() {
    %0 = firrtl.xmr.ref @XmrRef_path : !firrtl.rwprobe<uint<1>, @A>
    firrtl.layerblock @A {
      %a = firrtl.wire sym @a : !firrtl.uint<1>
      %c1_ui1 = firrtl.constant 1 : !firrtl.const.uint<1>
      firrtl.ref.force_initial %c1_ui1, %0, %c1_ui1 : !firrtl.const.uint<1>, !firrtl.rwprobe<uint<1>, @A>, !firrtl.const.uint<1>
    }
  }

  //===--------------------------------------------------------------------===//
  // Inline Layers
  //===--------------------------------------------------------------------===//

  // CHECK:      sv.macro.decl @layer$Inline
  // CHECK-NEXT: sv.macro.decl @layer$Inline$Inline
  // CHECK-NEXT: sv.macro.decl @layer$Inline$Bound$Inline
  // CHECK-NEXT: sv.macro.decl @layer$Bound$Inline
  firrtl.layer @Inline inline {
    firrtl.layer @Inline inline {}

    firrtl.layer @Bound bind {
      firrtl.layer @Inline inline {}
    }
  }

  firrtl.layer @Bound bind {
    firrtl.layer @Inline inline {
      firrtl.layer @Bound bind {}
    }
  }

  // CHECK:      firrtl.module private @ModuleWithInlineLayerBlocks_Inline_Bound() {
  // CHECK-NEXT:   %w3 = firrtl.wire : !firrtl.uint<3>
  // CHECK-NEXT: }

  // CHECK:      firrtl.module private @ModuleWithInlineLayerBlocks_Bound_Inline_Bound() {
  // CHECK-NEXT:   %w6 = firrtl.wire : !firrtl.uint<6>
  // CHECK-NEXT: }

  // CHECK:      firrtl.module private @ModuleWithInlineLayerBlocks_Bound() {
  // CHECK-NEXT:   %w4 = firrtl.wire
  // CHECK-NEXT:   sv.ifdef @layer$Bound$Inline {
  // CHECK-NEXT:     %w5 = firrtl.wire
  // CHECK-NEXT:   }
  // CHECK-NEXT: }

  // CHECK-NEXT: firrtl.module @ModuleWithInlineLayerBlocks() {
  // CHECK-NEXT:   sv.ifdef @layer$Inline {
  // CHECK-NEXT:     %w1 = firrtl.wire
  // CHECK-NEXT:     sv.ifdef @layer$Inline$Inline {
  // CHECK-NEXT:       %w2 = firrtl.wire
  // CHECK-NEXT:     }
  // CHECK-NEXT:     firrtl.instance inline_bound {{.*}} @ModuleWithInlineLayerBlocks_Inline_Bound()
  // CHECK-NEXT:   }
  // CHECK-NEXT:   firrtl.instance bound_inline_bound
  // CHECK-NEXT:   firrtl.instance bound {{.*}} @ModuleWithInlineLayerBlocks_Bound()
  // CHECK-NEXT: }
  firrtl.module @ModuleWithInlineLayerBlocks() {
    firrtl.layerblock @Inline {
      %w1 = firrtl.wire : !firrtl.uint<1>
      firrtl.layerblock @Inline::@Inline {
        %w2 = firrtl.wire : !firrtl.uint<2>
      }
      firrtl.layerblock @Inline::@Bound {
        %w3 = firrtl.wire : !firrtl.uint<3>
      }
    }

    firrtl.layerblock @Bound {
      %w4 = firrtl.wire : !firrtl.uint<4>
      firrtl.layerblock @Bound::@Inline {
        %w5 = firrtl.wire : !firrtl.uint<5>
        firrtl.layerblock @Bound::@Inline::@Bound {
          %w6 = firrtl.wire : !firrtl.uint<6>
        }
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
// CHECK: hw.hierpath private @[[Simple_A_B_cc_path:.+]] [@Simple::@a_b, @Simple_A_B::@[[Simple_A_B_cc_sym:.+]]]
//
// CHECK:      firrtl.module private @Simple_A_B_C() {
// CHECK-NEXT:   %0 = firrtl.xmr.deref @[[Simple_A_B_cc_path]]
// CHECK-NEXT:   %ccc = firrtl.node %0
// CHECK-NEXT: }
//
// CHECK:      hw.hierpath private @[[Simple_b_path:.+]] [@Simple::@[[Simple_b_sym:.+]]]
// CHECK-NEXT: hw.hierpath private @[[Simple_A_c_path:.+]] [@Simple::@a, @Simple_A::@[[Simple_A_c_sym:.+]]]
//
// CHECK:      firrtl.module private @Simple_A_B() {
// CHECK-NEXT:   %0 = firrtl.xmr.deref @[[Simple_A_c_path]]
// CHECK-NEXT:   %1 = firrtl.xmr.deref @[[Simple_b_path]]
// CHECK-NEXT:   %bb = firrtl.node %1
// CHECK-NEXT:   %cc = firrtl.node %0
// CHECK-NEXT:   %cc_0 = firrtl.node sym @[[Simple_A_B_cc_sym]] %cc {name = "cc"}
// CHECK-NEXT: }
//
// CHECK: hw.hierpath private @[[Simple_a_path:.+]] [@Simple::@[[Simple_a_sym:.+]]]
//
// CHECK:      firrtl.module private @Simple_A() {
// CHECK-NEXT:   %0 = firrtl.xmr.deref @[[Simple_a_path]]
// CHECK-NEXT:   %aa = firrtl.node %0
// CHECK-NEXT:   %c = firrtl.wire
// CHECK-NEXT:   %c_0 = firrtl.node sym @[[Simple_A_c_sym]] %c {name = "c"}
// CHECK-NEXT: }
//
// CHECK:      firrtl.module @Simple() {
// CHECK-NEXT:   %a = firrtl.wire
// CHECK-NEXT:   %a_0 = firrtl.node sym @[[Simple_a_sym]] %a {name = "a"}
// CHECK-NEXT:   %b = firrtl.wire
// CHECK-NEXT:   %b_1 = firrtl.node sym @[[Simple_b_sym]] %b {name = "b"}
// CHECK-NEXT:   firrtl.instance a_b_c sym @a_b_c {doNotPrint, output_file = #hw.output_file<"layers-Simple-A-B-C.sv", excludeFromFileList>} @Simple_A_B_C()
// CHECK-NEXT:   firrtl.instance a_b sym @a_b {doNotPrint, output_file = #hw.output_file<"layers-Simple-A-B.sv", excludeFromFileList>} @Simple_A_B()
// CHECK-NEXT:   firrtl.instance a sym @a {doNotPrint, output_file = #hw.output_file<"layers-Simple-A.sv", excludeFromFileList>} @Simple_A()
// CHECK-NEXT: }

// CHECK:      sv.macro.decl @layers_Simple_A["layers_Simple_A"]
// CHECK-NEXT: emit.file "layers-Simple-A.sv" {
// CHECK-NEXT:   sv.ifdef  @layers_Simple_A {
// CHECK-NEXT:   } else {
// CHECK-NEXT:     sv.macro.def @layers_Simple_A ""
// CHECK-NEXT:     firrtl.bind <@Simple::@a>
// CHECK-NEXT:   }
// CHECK-NEXT: }
// CHECK-NEXT: sv.macro.decl @layers_Simple_A_B["layers_Simple_A_B"]
// CHECK-NEXT: emit.file "layers-Simple-A-B.sv" {
// CHECK-NEXT:   sv.ifdef  @layers_Simple_A_B {
// CHECK-NEXT:   } else {
// CHECK-NEXT:     sv.macro.def @layers_Simple_A_B ""
// CHECK-NEXT:     sv.include  local "layers-Simple-A.sv"
// CHECK-NEXT:     firrtl.bind <@Simple::@a_b>
// CHECK-NEXT:   }
// CHECK-NEXT: }
// CHECK-NEXT: sv.macro.decl @layers_Simple_A_B_C["layers_Simple_A_B_C"]
// CHECK-NEXT: emit.file "layers-Simple-A-B-C.sv" {
// CHECK-NEXT:   sv.ifdef  @layers_Simple_A_B_C {
// CHECK-NEXT:   } else {
// CHECK-NEXT:     sv.macro.def @layers_Simple_A_B_C ""
// CHECK-NEXT:     sv.include  local "layers-Simple-A-B.sv"
// CHECK-NEXT:     firrtl.bind <@Simple::@a_b_c>
// CHECK-NEXT:   }
// CHECK-NEXT: }

// -----

firrtl.circuit "ModuleNameConflict" {
  firrtl.layer @A bind {}
  firrtl.module @ModuleNameConflict_A() {}
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
// CHECK:       firrtl.module @ModuleNameConflict_A()
// CHECK:       firrtl.module private @[[groupModule:[_a-zA-Z0-9_]+]]()
//
// CHECK:       firrtl.module @ModuleNameConflict()
// CHECK-NOT:   firrtl.module
// CHECK:         firrtl.instance foo @ModuleNameConflict_A()
// CHECK-NEXT:    firrtl.instance {{.+}} sym @{{.+}} {doNotPrint, {{.*}}}
// CHECK-SAME:      @[[groupModule]](

// -----
// Layerblock lowering must allow a value to be captured twice.
// https://github.com/llvm/circt/issues/6694

firrtl.circuit "CaptureHardwareMultipleTimes" {
  firrtl.layer @A bind {
    firrtl.layer @B bind {}
  }

  firrtl.extmodule @CaptureHardwareMultipleTimes ()

  // CHECK: hw.hierpath private @[[path:.+]] [@CaptureSrcTwice::@[[sym:.+]]]
  //
  // CHECK: firrtl.module private @[[A:.+]]()
  // CHECK:   %0 = firrtl.xmr.deref @[[path]] : !firrtl.uint<1>
  // CHECK:   %1 = firrtl.add %0, %0 : (!firrtl.uint<1>, !firrtl.uint<1>) -> !firrtl.uint<2>
  // CHECK: }
  // CHECK: firrtl.module @CaptureSrcTwice() {
  // CHECK:   %c0_ui1 = firrtl.constant 0 : !firrtl.uint<1>
  // CHECK:   %_layer_probe = firrtl.node sym @[[sym]] %c0_ui1 : !firrtl.uint<1>
  // CHECK:   firrtl.instance {{.+}} @[[A]]
  // CHECK: }
  firrtl.module @CaptureSrcTwice() {
    %c0_ui1 = firrtl.constant 0 : !firrtl.uint<1>
    firrtl.layerblock @A {
      %0 = firrtl.add %c0_ui1, %c0_ui1 : (!firrtl.uint<1>, !firrtl.uint<1>) -> !firrtl.uint<2>
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
  firrtl.layer @A bind {}
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
  firrtl.layer @A bind attributes {output_file = #hw.output_file<"testbench/", excludeFromFileList>} {}
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
// CHECK:       firrtl.module {{.*}} @Bar_A
// CHECK-SAME:    #hw.output_file<"testbench{{/|\\\\}}", excludeFromFileList>
// CHECK:       firrtl.module {{.*}} @Foo_A
// CHECK-SAME:    #hw.output_file<"testbench{{/|\\\\}}", excludeFromFileList>
// CHECK:       emit.file "testbench{{/|\\\\}}layers-Foo-A.sv"

// -----
// Check that we correctly implement the verilog header and footer for B.

firrtl.circuit "Foo" {
  firrtl.layer @A bind {
    firrtl.layer @X bind {}
  }
  firrtl.layer @B bind {}
  firrtl.module @Foo() {}
}

// CHECK: sv.macro.decl @layers_Foo_B["layers_Foo_B"]
// CHECK: emit.file "layers-Foo-B.sv" {
// CHECK:   sv.ifdef  @layers_Foo_B {
// CHECK:   } else {
// CHECK:     sv.macro.def @layers_Foo_B ""
// CHECK:   }
// CHECK: }

// -----

// Check sv.verbatim inner refs are updated, as occurs with views under layers.
// CHECK-LABEL: circuit "Verbatim"
firrtl.circuit "Verbatim" {
  firrtl.layer @ViewLayer bind { }
  firrtl.module @Verbatim() {
    firrtl.layerblock @ViewLayer {
      %c1_ui10 = firrtl.constant 1 : !firrtl.uint<10>
      %n = firrtl.node sym @node %c1_ui10 : !firrtl.uint<10>
      sv.verbatim "// node: {{0}}, {{1}}"(%n) : !firrtl.uint<10> {symbols = [#hw.innerNameRef<@Verbatim::@node>]}
    }
  }
}
// CHECK:        firrtl.module private @[[VL:.+]]() {
// CHECK-NEXT:     firrtl.constant 1
// CHECK-NEXT:     firrtl.node sym @node
// CHECK-NEXT:     sv.verbatim
// CHECK-SAME:     !firrtl.uint<10> {symbols = [#hw.innerNameRef<@[[VL]]::@node>]}
// CHECK-NEXT:   }

// -----

// Check that for public modules, bindfiles are generated for all known layers.
firrtl.circuit "Test" {
  firrtl.layer @A bind { }
  firrtl.layer @B bind { }
  firrtl.module public @Test() {}
  firrtl.module public @Test2() {}
}
// CHECK: emit.file "layers-Test-A.sv"
// CHECK: emit.file "layers-Test-B.sv"
// CHECK: emit.file "layers-Test2-A.sv"
// CHECK: emit.file "layers-Test2-B.sv"

// -----

// Check that for private modules, bindfiles files are only generated if they are used.
firrtl.circuit "Top" {
firrtl.layer @A bind { }
  firrtl.layer @B bind { }

  // CHECK:     emit.file "layers-Bottom-A.sv"
  // CHECK-NOT: emit.file "layers-Bottom-B.sv"
  firrtl.module private @Bottom() {
    firrtl.layerblock @A {}
  }

  // CHECK:     emit.file "layers-Middle-A.sv"
  // CHECK-NOT: emit.file "layers-Middle-B.sv"
  firrtl.module private @Middle() {
    firrtl.instance bottom @Bottom()
  }

  // CHECK:     emit.file "layers-Top-A.sv"
  // CHECK:     emit.file "layers-Top-B.sv"
  firrtl.module public @Top() {
    firrtl.instance middle @Middle()
  }
}

// -----

// Check that no bindfiles are created for inline layers.
firrtl.circuit "Top" {
  firrtl.layer @Inline inline {}

  firrtl.layer @Bound bind {
    firrtl.layer @Inline inline {}
  }

  // CHECK-NOT: emit.file "layers-Top-Inline.sv"
  // CHECK-NOT: emit.file "layers-Top-Bound-Inline.sv"
  firrtl.module @Top() {}
}

// -----

// Check that no duplicate include statements are generated.
firrtl.circuit "Top" {
  firrtl.layer @Layer bind {}

  firrtl.module @Component() {
    firrtl.layerblock @Layer {}
  }

  // There should only be one include statement.
  // CHECK:     emit.file "layers-Top-Layer.sv"
  // CHECK:     sv.include  local "layers-Component-Layer.sv"
  // CHECK-NOT: sv.include  local "layers-Component-Layer.sv"
  firrtl.module @Top() {
    firrtl.instance component1 @Component()
    firrtl.instance component2 @Component()
  }
}

// -----

// Check that only known bound-in layers are included.
firrtl.circuit "Top" {
  firrtl.layer @Layer bind {}

  firrtl.extmodule @ComponentA() attributes {knownLayers=[@Layer]}
  firrtl.extmodule @ComponentB() attributes {knownLayers=[]}
  // There should only be one include statement.
  // CHECK:     emit.file "layers-Top-Layer.sv"
  // CHECK:     sv.include local "layers-ComponentA-Layer.sv"
  // CHECK-NOT: sv.include local "layers-ComponentB-Layer.sv"
  firrtl.module @Top() {
    firrtl.instance componentA @ComponentA()
    firrtl.instance componentB @ComponentB()
  }
}

// -----

// When an extmodule knows of a child layer, it also knows the parent. Check
// that the parent bindfile is included. In this case, the child layer is
// inline, so it should NOT be included.
firrtl.circuit "Top" {
  firrtl.layer @ParentLayer bind {
    firrtl.layer @ChildLayer inline {}
  }

  firrtl.extmodule @Component() attributes {knownLayers=[@ParentLayer::@ChildLayer]}

  // There should only be one include statement.
  // CHECK:     emit.file "layers-Top-ParentLayer.sv"
  // CHECK:     sv.include local "layers-Component-ParentLayer.sv"
  // CHECK-NOT: sv.include local "layers-Component-ParentLayer-ChildLayer.sv"
  firrtl.module public @Top() {
    firrtl.instance component @Component()
  }
}

// -----

// When an extmodule knows of a child layer, it also knows the parent. Check
// that the parent bindfile is included. In this case, the child layer is
// bound-in, so it SHOULD be included too.
firrtl.circuit "Top" {
  firrtl.layer @ParentLayer bind {
    firrtl.layer @ChildLayer bind {}
  }

  firrtl.extmodule @Component() attributes {knownLayers=[@ParentLayer::@ChildLayer]}

  // CHECK: emit.file "layers-Top-ParentLayer.sv"
  // CHECK: sv.include local "layers-Component-ParentLayer.sv"
  // CHECK: sv.include local "layers-Component-ParentLayer-ChildLayer.sv"
  firrtl.module public @Top() {
    firrtl.instance component @Component()
  }
}

// -----

// Test that empty layer blocks do not create empty modules.
//
// See: https://github.com/llvm/circt/issues/9331
firrtl.circuit "EmptyLayerBlocks" {
  firrtl.layer @A bind {
    firrtl.layer @B bind {}
  }
  firrtl.layer @D inline {
    firrtl.layer @E bind {}
  }
  firrtl.module @EmptyLayerBlocks() {}

  // CHECK:     emit.file "layers-EmptyLeaf-A.sv"
  // CHECK:     emit.file "layers-EmptyLeaf-D-E.sv"
  // CHECK-NOT: @EmptyLeaf_A(
  // CHECK-NOT: @EmptyLeaf_D_E(
  firrtl.module @EmptyLeaf() {
    firrtl.layerblock @A {}
    firrtl.layerblock @D {
      %c = firrtl.wire : !firrtl.uint<1>
      firrtl.layerblock @D::@E {
      }
    }
  }

  // CHECK:     emit.file "layers-EmptyIntermediary-A.sv"
  // CHECK:     emit.file "layers-EmptyIntermediary-A-B.sv"
  // CHECK-NOT: @EmptyIntermediary_A(
  firrtl.module @EmptyIntermediary() {
    firrtl.layerblock @A {
      firrtl.layerblock @A::@B {
        %a_b = firrtl.wire : !firrtl.uint<1>
      }
    }
  }

  // CHECK:     emit.file "layers-EmptyAll-A.sv"
  // CHECK:     emit.file "layers-EmptyAll-A-B.sv"
  // CHECK-NOT: @EmptyAll_A(
  // CHECK-NOT: @EmptyAll_A_B(
  firrtl.module @EmptyAll() {
    firrtl.layerblock @A {
      firrtl.layerblock @A::@B {
      }
    }
  }
}
