// RUN: circt-opt %s -pass-pipeline='builtin.module(firrtl.circuit(any(firrtl-lower-objects)))' | FileCheck %s
// RUN: circt-opt %s -pass-pipeline='builtin.module(firrtl.circuit(any(firrtl-lower-objects,firrtl-lower-objects)))' | FileCheck %s
// RUN: circt-opt %s -pass-pipeline='builtin.module(firrtl.circuit(any(firrtl-lower-objects),firrtl-lower-classes))' | FileCheck %s --check-prefix=OM

firrtl.circuit "Partial" {
  firrtl.module @Partial() {}
  firrtl.class @Int(in %input: !firrtl.integer, out %output: !firrtl.integer) {
    firrtl.propassign %output, %input : !firrtl.integer
  }

  // Classes remain unchanged, even when all object inputs are constants.
  // CHECK-LABEL: firrtl.class @Hoisting
  // OM-LABEL: om.class @Hoisting
  firrtl.class @Hoisting() {
    // CHECK-NEXT: %[[BEFORE:.*]] = firrtl.integer 1
    // CHECK-NEXT: %obj = firrtl.object @Int
    // CHECK-NEXT: %[[IN:.*]] = firrtl.object.subfield %obj[input]
    // CHECK-NEXT: %[[AFTER:.*]] = firrtl.integer 2
    // CHECK-NEXT: %[[LAST:.*]] = firrtl.integer 3
    // CHECK-NEXT: firrtl.propassign %[[IN]], %[[AFTER]] : !firrtl.integer
    // CHECK-NEXT: }
    %before = firrtl.integer 1
    %obj = firrtl.object @Int(in input: !firrtl.integer, out output: !firrtl.integer)
    %in = firrtl.object.subfield %obj[input] : !firrtl.class<@Int(in input: !firrtl.integer, out output: !firrtl.integer)>
    %after = firrtl.integer 2
    %last = firrtl.integer 3
    firrtl.propassign %in, %after : !firrtl.integer
  }

  // Both dependent and constant-input objects in classes remain untouched.
  // CHECK-LABEL: firrtl.class @Mixed
  // OM-LABEL: om.class @Mixed
  firrtl.class @Mixed(out %out: !firrtl.integer) {
    // CHECK-NEXT: %a = firrtl.object @Int
    %a = firrtl.object @Int(in input: !firrtl.integer, out output: !firrtl.integer)
    // CHECK-NEXT: %[[AIN:.*]] = firrtl.object.subfield %a[input]
    %a_in = firrtl.object.subfield %a[input] : !firrtl.class<@Int(in input: !firrtl.integer, out output: !firrtl.integer)>
    // CHECK-NEXT: %[[AOUT:.*]] = firrtl.object.subfield %a[output]
    %a_out = firrtl.object.subfield %a[output] : !firrtl.class<@Int(in input: !firrtl.integer, out output: !firrtl.integer)>
    // CHECK-NEXT: firrtl.propassign %out, %[[AOUT]] : !firrtl.integer
    firrtl.propassign %out, %a_out : !firrtl.integer
    // CHECK-NEXT: %b = firrtl.object @Int
    %b = firrtl.object @Int(in input: !firrtl.integer, out output: !firrtl.integer)
    // CHECK-NEXT: %[[BIN:.*]] = firrtl.object.subfield %b[input]
    %b_in = firrtl.object.subfield %b[input] : !firrtl.class<@Int(in input: !firrtl.integer, out output: !firrtl.integer)>
    // CHECK-NEXT: %[[BOUT:.*]] = firrtl.object.subfield %b[output]
    %b_out = firrtl.object.subfield %b[output] : !firrtl.class<@Int(in input: !firrtl.integer, out output: !firrtl.integer)>
    // CHECK-NEXT: %[[C:.*]] = firrtl.integer 4
    %constant = firrtl.integer 4
    // CHECK-NEXT: firrtl.propassign %[[AIN]], %[[BOUT]] : !firrtl.integer
    // CHECK-NEXT: firrtl.propassign %[[BIN]], %[[C]] : !firrtl.integer
    // CHECK-NEXT: }
    firrtl.propassign %a_in, %b_out : !firrtl.integer
    firrtl.propassign %b_in, %constant : !firrtl.integer
  }

  // Module expressions and their constants are hoisted together.
  // CHECK-LABEL: firrtl.module @Expression
  // OM-LABEL: om.class @Expression_Class
  firrtl.module @Expression() {
    %obj = firrtl.object @Int(in input: !firrtl.integer, out output: !firrtl.integer)
    %in = firrtl.object.subfield %obj[input] : !firrtl.class<@Int(in input: !firrtl.integer, out output: !firrtl.integer)>
    // CHECK-NEXT: %[[C:.*]] = firrtl.integer 5
    %constant = firrtl.integer 5
    // CHECK-NEXT: %[[SUM:.*]] = firrtl.integer.add %[[C]], %[[C]]
    %sum = firrtl.integer.add %constant, %constant : (!firrtl.integer, !firrtl.integer) -> !firrtl.integer
    // CHECK-NEXT: %obj = firrtl.object.create(%[[SUM]])
    // CHECK-NEXT: }
    firrtl.propassign %in, %sum : !firrtl.integer
  }

  // Self-cyclic inputs remain in declaration form in both modules and classes.
  // CHECK-LABEL: firrtl.module @SelfModule
  // OM-LABEL: om.class @SelfModule_Class
  firrtl.module @SelfModule() {
    // CHECK-NEXT: %obj = firrtl.object @Int
    %obj = firrtl.object @Int(in input: !firrtl.integer, out output: !firrtl.integer)
    // CHECK-NEXT: %[[IN:.*]] = firrtl.object.subfield %obj[input]
    %in = firrtl.object.subfield %obj[input] : !firrtl.class<@Int(in input: !firrtl.integer, out output: !firrtl.integer)>
    // CHECK-NEXT: %[[OUT:.*]] = firrtl.object.subfield %obj[output]
    %out = firrtl.object.subfield %obj[output] : !firrtl.class<@Int(in input: !firrtl.integer, out output: !firrtl.integer)>
    // CHECK-NEXT: firrtl.propassign %[[IN]], %[[OUT]] : !firrtl.integer
    // CHECK-NEXT: }
    firrtl.propassign %in, %out : !firrtl.integer
  }
  // CHECK-LABEL: firrtl.class @SelfClass
  // OM-LABEL: om.class @SelfClass
  firrtl.class @SelfClass() {
    // CHECK-NEXT: %obj = firrtl.object @Int
    %obj = firrtl.object @Int(in input: !firrtl.integer, out output: !firrtl.integer)
    // CHECK-NEXT: %[[IN:.*]] = firrtl.object.subfield %obj[input]
    %in = firrtl.object.subfield %obj[input] : !firrtl.class<@Int(in input: !firrtl.integer, out output: !firrtl.integer)>
    // CHECK-NEXT: %[[OUT:.*]] = firrtl.object.subfield %obj[output]
    %out = firrtl.object.subfield %obj[output] : !firrtl.class<@Int(in input: !firrtl.integer, out output: !firrtl.integer)>
    // CHECK-NEXT: firrtl.propassign %[[IN]], %[[OUT]] : !firrtl.integer
    // CHECK-NEXT: }
    firrtl.propassign %in, %out : !firrtl.integer
  }
}
