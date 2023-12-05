// RUN: circt-opt -pass-pipeline="builtin.module(firrtl.circuit(firrtl.module(firrtl-layer-sink)))" %s | FileCheck %s

// Test that simple things with uses only in layers are sunk.
//
// CHECK-LABEL: firrtl.circuit "SimpleSink"
firrtl.circuit "SimpleSink" {
  firrtl.layer @Constant bind {}
  firrtl.layer @Node bind {}
  firrtl.layer @Expression bind {}
  firrtl.layer @Wire bind {}
  firrtl.layer @Reg bind {}
  firrtl.layer @Mem bind {}
  firrtl.layer @Instance bind {}
  firrtl.layer @DeepSinking bind {
    firrtl.layer @A bind {
      firrtl.layer @B bind {
        firrtl.layer @C bind {
        }
      }
    }
  }
  firrtl.layer @RegCycle bind {}
  firrtl.layer @InstanceCycle bind {}
  firrtl.layer @BlockedSinking bind {
    firrtl.layer @A bind {}
    firrtl.layer @B bind {}
  }
  firrtl.layer @DontTouch bind {}
  firrtl.layer @Aggregates bind {}

  firrtl.module @Foo(in %a: !firrtl.uint<1>, out %b: !firrtl.uint<1>) {
    firrtl.strictconnect %b, %a : !firrtl.uint<1>
  }

  // CHECK: firrtl.module @SimpleSink
  firrtl.module @SimpleSink(in %clock: !firrtl.clock, in %a: !firrtl.uint<1>) {

    //===------------------------------------------------------------------===//
    // Check that various kinds of operations are all sunk.
    //===------------------------------------------------------------------===//
    %c0_ui1 = firrtl.constant 0 : !firrtl.uint<1>
    firrtl.layerblock @Constant {
      %layer_constant = firrtl.node %c0_ui1 : !firrtl.uint<1>
    }
    // CHECK-NEXT: firrtl.layerblock @Constant {
    // CHECK-NEXT:   %c0_ui1 = firrtl.constant 0
    // CHECK-NEXT:   %layer_constant = firrtl.node %c0_ui1
    // CHECK-NEXT: }

    %node = firrtl.node %a : !firrtl.uint<1>
    firrtl.layerblock @Node {
      %layer_node = firrtl.node %node : !firrtl.uint<1>
    }
    // CHECK-NEXT: firrtl.layerblock @Node {
    // CHECK-NEXT:   %node = firrtl.node %a
    // CHECK-NEXT:   %layer_node = firrtl.node %node
    // CHECK-NEXT: }

    %0 = firrtl.not %a : (!firrtl.uint<1>) -> !firrtl.uint<1>
    firrtl.layerblock @Expression {
      %layer_expression = firrtl.node %0 : !firrtl.uint<1>
    }
    // CHECK-NEXT: firrtl.layerblock @Expression {
    // CHECK-NEXT:   %0 = firrtl.not %a
    // CHECK-NEXT:   %layer_expression = firrtl.node %0
    // CHECK-NEXT: }

    %wire = firrtl.wire : !firrtl.uint<1>
    firrtl.strictconnect %wire, %a : !firrtl.uint<1>
    firrtl.layerblock @Wire {
      %layer_wire = firrtl.node %wire : !firrtl.uint<1>
    }
    // CHECK-NEXT: firrtl.layerblock @Wire {
    // CHECK-NEXT:   %wire = firrtl.wire
    // CHECK-NEXT:   firrtl.strictconnect %wire, %a
    // CHECK-NEXT:   %layer_wire = firrtl.node %wire
    // CHECK-NEXT: }

    %reg = firrtl.reg %clock : !firrtl.clock, !firrtl.uint<1>
    firrtl.strictconnect %reg, %a : !firrtl.uint<1>
    firrtl.layerblock @Reg {
      %layeyr_reg = firrtl.node %reg : !firrtl.uint<1>
    }
    // CHECK-NEXT: firrtl.layerblock @Reg {
    // CHECK-NEXT:   %reg = firrtl.reg %clock
    // CHECK-NEXT:   firrtl.strictconnect %reg, %a
    // CHECK-NEXT:   %layeyr_reg = firrtl.node %reg
    // CHECK-NEXT: }

    %mem_r = firrtl.mem Undefined {depth = 2, name = "mem", portNames = ["r"], readLatency = 0 : i32, writeLatency = 1 : i32} : !firrtl.bundle<addr: uint<1>, en: uint<1>, clk: clock, data flip: uint<8>>
    %mem_r_data = firrtl.subfield %mem_r[data] : !firrtl.bundle<addr: uint<1>, en: uint<1>, clk: clock, data flip: uint<8>>
    firrtl.layerblock @Mem {
      %layer_mem = firrtl.node %mem_r_data : !firrtl.uint<8>
    }
    // CHECK-NEXT: firrtl.layerblock @Mem {
    // CHECK-NEXT:   %mem_r = firrtl.mem
    // CHECK-NEXT:   %[[mem_r_data:[_A-Za-z0-9]+]] = firrtl.subfield %mem_r[data]
    // CHECK-NEXT:   %layer_mem = firrtl.node %[[mem_r_data]]
    // CHECK-NEXT: }

    %foo_a, %foo_b = firrtl.instance foo @Foo(in a: !firrtl.uint<1>, out b: !firrtl.uint<1>)
    firrtl.strictconnect %foo_a, %a : !firrtl.uint<1>
    firrtl.layerblock @Instance {
      %layer_instance = firrtl.node %foo_b : !firrtl.uint<1>
    }
    // CHECK-NEXT: firrtl.layerblock @Instance {
    // CHECK-NEXT:   %foo_a, %foo_b = firrtl.instance foo @Foo
    // CHECK-NEXT:   firrtl.strictconnect %foo_a, %a
    // CHECK-NEXT:   %layer_instance = firrtl.node %foo_b
    // CHECK-NEXT: }

    //===------------------------------------------------------------------===//
    // Check that sinking sinks to the highest layer.
    //===------------------------------------------------------------------===//
    %node_deep = firrtl.node %a : !firrtl.uint<1>
    firrtl.layerblock @DeepSinking {
      %node_deep_0 = firrtl.node %node_deep : !firrtl.uint<1>
      firrtl.layerblock @DeepSinking::@A {
        %node_deep_1 = firrtl.node %node_deep_0 : !firrtl.uint<1>
        firrtl.layerblock @DeepSinking::@A::@B {
          %node_deep_2 = firrtl.node %node_deep_1 : !firrtl.uint<1>
          firrtl.layerblock @DeepSinking::@A::@B::@C {
            %node_deep_3 = firrtl.node %node_deep_2 : !firrtl.uint<1>
          }
        }
      }
    }
    // CHECK-NEXT: firrtl.layerblock @DeepSinking {
    // CHECK-NEXT:   firrtl.layerblock @DeepSinking::@A {
    // CHECK-NEXT:     firrtl.layerblock @DeepSinking::@A::@B {
    // CHECK-NEXT:       firrtl.layerblock @DeepSinking::@A::@B::@C {
    // CHECK-NEXT:         %node_deep = firrtl.node %a : !firrtl.uint<1>
    // CHECK-NEXT:         %node_deep_0 = firrtl.node %node_deep : !firrtl.uint<1>
    // CHECK-NEXT:         %node_deep_1 = firrtl.node %node_deep_0 : !firrtl.uint<1>
    // CHECK-NEXT:         %node_deep_2 = firrtl.node %node_deep_1 : !firrtl.uint<1>
    // CHECK-NEXT:         %node_deep_3 = firrtl.node %node_deep_2 : !firrtl.uint<1>
    // CHECK-NEXT:       }
    // CHECK-NEXT:     }
    // CHECK-NEXT:   }
    // CHECK-NEXT: }

    //===------------------------------------------------------------------===//
    // Check that operations with cycles are collectively sunk.
    //===------------------------------------------------------------------===//
    %reg_cycle_a = firrtl.reg %clock : !firrtl.clock, !firrtl.uint<1>
    %reg_cycle_b = firrtl.reg %clock : !firrtl.clock, !firrtl.uint<1>
    firrtl.strictconnect %reg_cycle_a, %reg_cycle_b : !firrtl.uint<1>
    firrtl.strictconnect %reg_cycle_b, %reg_cycle_a : !firrtl.uint<1>
    firrtl.layerblock @RegCycle {
      %layer_reg_cycle = firrtl.node %reg_cycle_b : !firrtl.uint<1>
    }
    // CHECK-NEXT: firrtl.layerblock @RegCycle {
    // CHECK-NEXT:   %reg_cycle_a = firrtl.reg %clock
    // CHECK-NEXT:   %reg_cycle_b = firrtl.reg %clock
    // CHECK-NEXT:   firrtl.strictconnect %reg_cycle_a, %reg_cycle_b
    // CHECK-NEXT:   firrtl.strictconnect %reg_cycle_b, %reg_cycle_a
    // CHECK-NEXT:   %layer_reg_cycle = firrtl.node %reg_cycle_b
    // CHECK-NEXT: }

    %bar_a, %bar_b = firrtl.instance bar @Foo(in a: !firrtl.uint<1>, out b: !firrtl.uint<1>)
    %baz_a, %baz_b = firrtl.instance baz @Foo(in a: !firrtl.uint<1>, out b: !firrtl.uint<1>)
    firrtl.strictconnect %bar_a, %baz_b : !firrtl.uint<1>
    firrtl.strictconnect %baz_a, %bar_b : !firrtl.uint<1>
    firrtl.layerblock @InstanceCycle {
      %layer_instance_cycle = firrtl.node %baz_b : !firrtl.uint<1>
    }
    // CHECK-NEXT: firrtl.layerblock @InstanceCycle {
    // CHECK-NEXT:   %bar_a, %bar_b = firrtl.instance bar @Foo(in a: !firrtl.uint<1>, out b: !firrtl.uint<1>)
    // CHECK-NEXT:   %baz_a, %baz_b = firrtl.instance baz @Foo(in a: !firrtl.uint<1>, out b: !firrtl.uint<1>)
    // CHECK-NEXT:   firrtl.strictconnect %bar_a, %baz_b : !firrtl.uint<1>
    // CHECK-NEXT:   firrtl.strictconnect %baz_a, %bar_b : !firrtl.uint<1>
    // CHECK-NEXT:   %layer_instance_cycle = firrtl.node %baz_b : !firrtl.uint<1>
    // CHECK-NEXT: }

    //===------------------------------------------------------------------===//
    // Check that shared usage blocks sinking.
    //===------------------------------------------------------------------===//
    firrtl.layerblock @BlockedSinking {
      %1 = firrtl.node %a : !firrtl.uint<1>
      firrtl.layerblock @BlockedSinking::@A {
        %2 = firrtl.node %1 : !firrtl.uint<1>
      }
      firrtl.layerblock @BlockedSinking::@B {
        %2 = firrtl.node %1 : !firrtl.uint<1>
      }
    }
    // CHECK-NEXT: firrtl.layerblock @BlockedSinking {
    // CHECK-NEXT:   %[[shared:[_A-Za-z0-9]+]] = firrtl.node %a
    // CHECK-NEXT:   firrtl.layerblock @BlockedSinking::@A {
    // CHECK-NEXT:     firrtl.node %[[shared]]
    // CHECK-NEXT:   }
    // CHECK-NEXT:   firrtl.layerblock @BlockedSinking::@B {
    // CHECK-NEXT:     firrtl.node %[[shared]]
    // CHECK-NEXT:   }
    // CHECK-NEXT: }

    //===------------------------------------------------------------------===//
    // Check annotation interactions.
    //===------------------------------------------------------------------===//
    %node_dontTouch = firrtl.node %a {annotations = [{class = "firrtl.transforms.DontTouchAnnotation"}]} : !firrtl.uint<1>
    firrtl.layerblock @BlockedSinking {
      %layer_node_dontTouch = firrtl.node %node_dontTouch : !firrtl.uint<1>
    }
    // CHECK-NEXT: %node_dontTouch = firrtl.node %a
    // CHECK-NEXT: firrtl.layerblock @BlockedSinking {
    // CHECK-NEXT:   %layer_node_dontTouch = firrtl.node %node_dontTouch
    // CHECK-NEXT: }
  }
}
