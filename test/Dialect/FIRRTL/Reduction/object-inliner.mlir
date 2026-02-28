// UNSUPPORTED: system-windows
//   See https://github.com/llvm/circt/issues/4129
// RUN: circt-reduce %s --test /usr/bin/env --test-arg true --keep-best=0 --include firrtl-object-inliner | FileCheck %s

// Test basic `ObjectOp` inlining functionality.
firrtl.circuit "ObjectInlinerTest" {
  firrtl.extmodule @ObjectInlinerTest()

  // Simple class with input and output ports (using property types)
  firrtl.class @SimpleClass(in %input: !firrtl.string, out %output: !firrtl.string) {
    // Add debug variable operation on the input port
    dbg.variable "class_input", %input : !firrtl.string
    firrtl.propassign %output, %input : !firrtl.string
  }

  // CHECK-LABEL: firrtl.class @SimpleTest
  firrtl.class @SimpleTest(in %in: !firrtl.string, out %out: !firrtl.string) {
    // CHECK-NOT: firrtl.object
    // CHECK-NOT: firrtl.object.subfield
    // CHECK: dbg.variable "class_input", %in : !firrtl.string
    // CHECK: firrtl.propassign %out, %in : !firrtl.string
    %obj = firrtl.object @SimpleClass(in input: !firrtl.string, out output: !firrtl.string)
    %obj_input = firrtl.object.subfield %obj[input] : !firrtl.class<@SimpleClass(in input: !firrtl.string, out output: !firrtl.string)>
    %obj_output = firrtl.object.subfield %obj[output] : !firrtl.class<@SimpleClass(in input: !firrtl.string, out output: !firrtl.string)>

    firrtl.propassign %obj_input, %in : !firrtl.string
    firrtl.propassign %out, %obj_output : !firrtl.string
  }

  // Class with multiple ports (using property types)
  firrtl.class @MultiPortClass(in %a: !firrtl.string, in %b: !firrtl.string, out %result: !firrtl.string) {
    // Add debug variable operations on the input ports
    dbg.variable "input_a", %a : !firrtl.string
    dbg.variable "input_b", %b : !firrtl.string
    // Simple pass-through for demonstration
    firrtl.propassign %result, %a : !firrtl.string
  }

  // CHECK-LABEL: firrtl.class @MultiPortTest
  firrtl.class @MultiPortTest(in %x: !firrtl.string, in %y: !firrtl.string, out %result: !firrtl.string) {
    // CHECK-NOT: firrtl.object
    // CHECK-NOT: firrtl.object.subfield
    // CHECK: dbg.variable "input_a", %x : !firrtl.string
    // CHECK: dbg.variable "input_b", %y : !firrtl.string
    // CHECK: firrtl.propassign %result, %x : !firrtl.string
    %obj = firrtl.object @MultiPortClass(in a: !firrtl.string, in b: !firrtl.string, out result: !firrtl.string)
    %obj_a = firrtl.object.subfield %obj[a] : !firrtl.class<@MultiPortClass(in a: !firrtl.string, in b: !firrtl.string, out result: !firrtl.string)>
    %obj_b = firrtl.object.subfield %obj[b] : !firrtl.class<@MultiPortClass(in a: !firrtl.string, in b: !firrtl.string, out result: !firrtl.string)>
    %obj_result = firrtl.object.subfield %obj[result] : !firrtl.class<@MultiPortClass(in a: !firrtl.string, in b: !firrtl.string, out result: !firrtl.string)>

    firrtl.propassign %obj_a, %x : !firrtl.string
    firrtl.propassign %obj_b, %y : !firrtl.string
    firrtl.propassign %result, %obj_result : !firrtl.string
  }

  // Test case that creates a use-before-def after inlining the class.
  // CHECK-LABEL: firrtl.class @DominanceTest
  firrtl.class @DominanceTest(in %in: !firrtl.string) {
    // CHECK-NOT: firrtl.object
    // CHECK-NOT: firrtl.object.subfield
    %obj = firrtl.object @DominanceClass(in input: !firrtl.string)
    %obj_input = firrtl.object.subfield %obj[input] : !firrtl.class<@DominanceClass(in input: !firrtl.string)>

    // CHECK: [[TMP:%.+]] = firrtl.string "foo"
    // CHECK: dbg.variable "use", [[TMP]]
    %0 = firrtl.string "foo"
    firrtl.propassign %obj_input, %0 : !firrtl.string
  }

  firrtl.class @DominanceClass(in %input: !firrtl.string) {
    dbg.variable "use", %input : !firrtl.string
  }
}
