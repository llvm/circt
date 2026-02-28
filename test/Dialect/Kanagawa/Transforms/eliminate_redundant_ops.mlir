// RUN: circt-opt --split-input-file --pass-pipeline="builtin.module(kanagawa.design(kanagawa.container(kanagawa-eliminate-redundant-ops)))" %s | FileCheck %s

kanagawa.design @TestDesign {

// Test case 1: Eliminate redundant GetPortOps for input ports
// CHECK-LABEL: kanagawa.container sym @RedundantInputGetPort {
// CHECK-COUNT-1: kanagawa.get_port %{{.*}}, @in
// CHECK-NOT: kanagawa.get_port %{{.*}}, @in
kanagawa.container sym @RedundantInputGetPort {
  %instance1 = kanagawa.container.instance @child, <@TestDesign::@ChildContainer>

  // First GetPortOp for input port - should be kept
  %port_ref1 = kanagawa.get_port %instance1, @in : !kanagawa.scoperef<@TestDesign::@ChildContainer> -> !kanagawa.portref<in i32>

  // Second GetPortOp for same input port - should be eliminated
  %port_ref2 = kanagawa.get_port %instance1, @in : !kanagawa.scoperef<@TestDesign::@ChildContainer> -> !kanagawa.portref<in i32>

  // Third GetPortOp for same input port - should be eliminated
  %port_ref3 = kanagawa.get_port %instance1, @in : !kanagawa.scoperef<@TestDesign::@ChildContainer> -> !kanagawa.portref<in i32>
}

kanagawa.container sym @ChildContainer {
  %in = kanagawa.port.input "in" sym @in : i32
}

}

// -----

kanagawa.design @TestDesign2 {

// Test case 2: Eliminate redundant GetPortOps for output ports and port reads.
// CHECK-LABEL: kanagawa.container sym @RedundantOutputGetPort {
// CHECK-COUNT-1: kanagawa.get_port %{{.*}}, @out
// CHECK-NOT: kanagawa.get_port %{{.*}}, @out
// CHECK-COUNT-1: kanagawa.port.read %{{.*}} : !kanagawa.portref<out i32>
// CHECK-NOT: kanagawa.port.read %{{.*}} : !kanagawa.portref<out i32>
kanagawa.container sym @RedundantOutputGetPort {
  %instance1 = kanagawa.container.instance @child, <@TestDesign2::@ChildContainer>

  // First GetPortOp for output port - should be kept
  %port_ref1 = kanagawa.get_port %instance1, @out : !kanagawa.scoperef<@TestDesign2::@ChildContainer> -> !kanagawa.portref<out i32>

  // Second GetPortOp for same output port - should be eliminated
  %port_ref2 = kanagawa.get_port %instance1, @out : !kanagawa.scoperef<@TestDesign2::@ChildContainer> -> !kanagawa.portref<out i32>

  %val1 = kanagawa.port.read %port_ref1 : !kanagawa.portref<out i32>
  %val2 = kanagawa.port.read %port_ref2 : !kanagawa.portref<out i32>

  %sum = arith.addi %val1, %val2 : i32
}

kanagawa.container sym @ChildContainer {
  %in = kanagawa.port.input "in" sym @in : i32
  %out = kanagawa.port.output "out" sym @out : i32
  %const = hw.constant 100 : i32
  kanagawa.port.write %out, %const : !kanagawa.portref<out i32>
}

}
