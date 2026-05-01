// RUN: circt-opt --firrtl-lower-layers %s | FileCheck %s

// Test that RWProbe values captured in layer blocks are converted to XMR operations:
// - Instance ports use xmr.remote
// - Wires/registers use xmr.ref with hierarchical paths

// CHECK-LABEL: firrtl.circuit "RWProbeFromExtModule"
firrtl.circuit "RWProbeFromExtModule" {
  firrtl.layer @A bind {
  }

  // CHECK: firrtl.module private @RWProbeFromExtModule_A
  // CHECK-NEXT: %[[REMOTE:.+]] = firrtl.xmr.remote @RWProbeFromExtModule::@[[SYM:.+]], 1 : !firrtl.rwprobe<uint<1>>
  // CHECK-NEXT: %[[C1:.+]] = firrtl.constant 1
  // CHECK-NEXT: firrtl.ref.force_initial %[[C1]], %[[REMOTE]], %[[C1]]

  // CHECK: firrtl.module @RWProbeFromExtModule
  firrtl.module @RWProbeFromExtModule(in %clock: !firrtl.clock) attributes {convention = #firrtl<convention scalarized>} {
    // CHECK: %bar_probe, %bar_rwprobe = firrtl.instance bar sym @[[SYM]]
    %bar_probe, %bar_rwprobe = firrtl.instance bar @Bar(out probe: !firrtl.probe<uint<1>>, out rwprobe: !firrtl.rwprobe<uint<1>>)
    firrtl.layerblock @A {
      %c1_ui1 = firrtl.constant 1 : !firrtl.uint<1>
      
      firrtl.ref.force_initial %c1_ui1, %bar_rwprobe, %c1_ui1 : !firrtl.uint<1>, !firrtl.rwprobe<uint<1>>, !firrtl.uint<1>
    }
  }
  
  firrtl.extmodule private @Bar(out probe: !firrtl.probe<uint<1>>, out rwprobe: !firrtl.rwprobe<uint<1>> sym @probePort) attributes {convention = #firrtl<convention scalarized>}
}

// -----

// Test with local wire that produces RWProbe
// CHECK-LABEL: firrtl.circuit "RWProbeFromWire"
firrtl.circuit "RWProbeFromWire" {
  firrtl.layer @A bind {
  }

  // CHECK: hw.hierpath private @[[PATH:.+]] [@RWProbeFromWire::@[[SYM:.+]]]
  // CHECK: firrtl.module private @RWProbeFromWire_A
  // CHECK-NEXT: %[[XMRREF:.+]] = firrtl.xmr.ref @[[PATH]] : !firrtl.rwprobe<uint<8>>
  // CHECK-NEXT: %[[C1:.+]] = firrtl.constant 1
  // CHECK-NEXT: %[[C42:.+]] = firrtl.constant 42
  // CHECK-NEXT: firrtl.ref.force_initial %[[C1]], %[[XMRREF]], %[[C42]]

  // CHECK: firrtl.module @RWProbeFromWire
  firrtl.module @RWProbeFromWire(in %clock: !firrtl.clock) {
    // CHECK: %w, %w_ref = firrtl.wire sym @[[SYM]] forceable
    %w, %w_ref = firrtl.wire forceable : !firrtl.uint<8>, !firrtl.rwprobe<uint<8>>
    
    firrtl.layerblock @A {
      %c1 = firrtl.constant 1 : !firrtl.uint<1>
      %c42 = firrtl.constant 42 : !firrtl.uint<8>
      
      firrtl.ref.force_initial %c1, %w_ref, %c42 : !firrtl.uint<1>, !firrtl.rwprobe<uint<8>>, !firrtl.uint<8>
    }
  }
}

// -----

// Test with RWProbe from regular module port (uses xmr.remote for instance ports)
// CHECK-LABEL: firrtl.circuit "RWProbeFromModulePort"
firrtl.circuit "RWProbeFromModulePort" {
  firrtl.layer @A bind {
  }

  // CHECK: firrtl.module private @RWProbeFromModulePort_A
  // CHECK-NEXT: %[[REMOTE:.+]] = firrtl.xmr.remote @RWProbeFromModulePort::@[[SYM:.+]], 0 : !firrtl.rwprobe<uint<4>>
  // CHECK-NEXT: %[[C1:.+]] = firrtl.constant 1
  // CHECK-NEXT: %[[C5:.+]] = firrtl.constant 5
  // CHECK-NEXT: firrtl.ref.force_initial %[[C1]], %[[REMOTE]], %[[C5]]

  // CHECK: firrtl.module @RWProbeFromModulePort
  firrtl.module @RWProbeFromModulePort(in %clock: !firrtl.clock) {
    // CHECK: %child_out = firrtl.instance child sym @[[SYM]]
    %child_out = firrtl.instance child @Child(out out: !firrtl.rwprobe<uint<4>>)
    
    firrtl.layerblock @A {
      %c1 = firrtl.constant 1 : !firrtl.uint<1>
      %c5 = firrtl.constant 5 : !firrtl.uint<4>
      
      firrtl.ref.force_initial %c1, %child_out, %c5 : !firrtl.uint<1>, !firrtl.rwprobe<uint<4>>, !firrtl.uint<4>
    }
  }
  
  firrtl.module private @Child(out %out: !firrtl.rwprobe<uint<4>> sym @outPort) {
    %w, %w_ref = firrtl.wire forceable : !firrtl.uint<4>, !firrtl.rwprobe<uint<4>>
    firrtl.ref.define %out, %w_ref : !firrtl.rwprobe<uint<4>>
  }
}
