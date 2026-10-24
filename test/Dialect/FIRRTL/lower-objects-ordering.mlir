// RUN: circt-opt %s -pass-pipeline='builtin.module(firrtl.circuit(any(firrtl-lower-objects)))' | FileCheck %s
// RUN: circt-opt %s -pass-pipeline='builtin.module(firrtl.circuit(any(firrtl-lower-objects,firrtl-lower-objects)))' | FileCheck %s
// RUN: circt-opt %s -pass-pipeline='builtin.module(firrtl.circuit(any(firrtl-lower-objects)))' > %t
// RUN: circt-opt %t -pass-pipeline='builtin.module(firrtl.circuit(any(firrtl-lower-objects)))' | diff %t -
// RUN: circt-opt %s -pass-pipeline='builtin.module(firrtl.circuit(any(firrtl-lower-objects),firrtl-lower-classes))' | FileCheck %s --check-prefix=OM

firrtl.circuit "Ordering" {
  firrtl.module @Ordering() {}
  firrtl.class @Int(in %input: !firrtl.integer, out %output: !firrtl.integer) {
    firrtl.propassign %output, %input : !firrtl.integer
  }

  firrtl.extclass @Empty()
  firrtl.class @Holder(in %value: !firrtl.class<@Empty()>) {}

  // An input may be an object itself rather than one of its output fields.
  // Read assignment sources after replacing dependent declarations.
  // CHECK-LABEL: firrtl.module @ObjectInput
  // OM-LABEL: om.class @ObjectInput_Class
  firrtl.module @ObjectInput() {
    // CHECK-NEXT: %value = firrtl.object.create()
    // CHECK-NEXT: %holder = firrtl.object.create(%value)
    // CHECK-NEXT: }
    %holder = firrtl.object @Holder(in value: !firrtl.class<@Empty()>)
    %in = firrtl.object.subfield %holder[value] : !firrtl.class<@Holder(in value: !firrtl.class<@Empty()>)>
    %value = firrtl.object @Empty()
    firrtl.propassign %in, %value : !firrtl.class<@Empty()>
    // OM: %[[VALUE:.*]] = om.object @Empty
    // OM: om.object @Holder(%{{.*}}, %[[VALUE]])
  }

  // Convert a forward chain in dependency order. The shared expression is
  // hoisted once, and dependent declarations retain their attributes.
  // CHECK-LABEL: firrtl.module @Chain
  // OM-LABEL: om.class @Chain_Class
  firrtl.module @Chain(out %out: !firrtl.integer) {
    // CHECK-NEXT: %[[C:.*]] = firrtl.integer 7
    // CHECK-NEXT: %[[SUM:.*]] = firrtl.integer.add %[[C]], %[[C]]
    // CHECK-NEXT: %c = firrtl.object.create(%[[SUM]]) {test.attr = "dependency"}
    // CHECK-NEXT: %[[COUT:.*]] = firrtl.object.subfield %c[output]
    // CHECK-NEXT: %b = firrtl.object.create(%[[COUT]])
    // CHECK-NEXT: %[[BOUT:.*]] = firrtl.object.subfield %b[output]
    // CHECK-NEXT: %a = firrtl.object.create(%[[BOUT]])
    %a = firrtl.object @Int(in input: !firrtl.integer, out output: !firrtl.integer)
    %b = firrtl.object @Int(in input: !firrtl.integer, out output: !firrtl.integer)
    %c = firrtl.object {test.attr = "dependency"} @Int(in input: !firrtl.integer, out output: !firrtl.integer)
    %a_in = firrtl.object.subfield %a[input] : !firrtl.class<@Int(in input: !firrtl.integer, out output: !firrtl.integer)>
    %b_in = firrtl.object.subfield %b[input] : !firrtl.class<@Int(in input: !firrtl.integer, out output: !firrtl.integer)>
    %c_in = firrtl.object.subfield %c[input] : !firrtl.class<@Int(in input: !firrtl.integer, out output: !firrtl.integer)>
    // CHECK-NEXT: %[[AOUT:.*]] = firrtl.object.subfield %a[output]
    %a_out = firrtl.object.subfield %a[output] : !firrtl.class<@Int(in input: !firrtl.integer, out output: !firrtl.integer)>
    %b_out = firrtl.object.subfield %b[output] : !firrtl.class<@Int(in input: !firrtl.integer, out output: !firrtl.integer)>
    %c_out = firrtl.object.subfield %c[output] : !firrtl.class<@Int(in input: !firrtl.integer, out output: !firrtl.integer)>
    %constant = firrtl.integer 7
    %sum = firrtl.integer.add %constant, %constant : (!firrtl.integer, !firrtl.integer) -> !firrtl.integer
    firrtl.propassign %a_in, %b_out : !firrtl.integer
    firrtl.propassign %b_in, %c_out : !firrtl.integer
    firrtl.propassign %c_in, %sum : !firrtl.integer
    // CHECK-NEXT: firrtl.propassign %out, %[[AOUT]]
    firrtl.propassign %out, %a_out : !firrtl.integer
    // CHECK-NEXT: %shared = firrtl.object.create(%[[SUM]])
    // CHECK-NEXT: }
    %shared = firrtl.object @Int(in input: !firrtl.integer, out output: !firrtl.integer)
    %shared_in = firrtl.object.subfield %shared[input] : !firrtl.class<@Int(in input: !firrtl.integer, out output: !firrtl.integer)>
    firrtl.propassign %shared_in, %sum : !firrtl.integer
    // OM: om.object @Int
    // OM: om.object @Int
    // OM: om.object @Int
    // OM: om.object @Int
  }

  // A failed plan must not move even its acyclic prerequisites. The cycle
  // does not prevent lowering a later independent object or a consumer whose
  // input already dominates its declaration.
  // CHECK-LABEL: firrtl.module @Partial
  // OM-LABEL: om.class @Partial_Class
  firrtl.module @Partial() {
    // CHECK-NEXT: %cycle = firrtl.object {test.attr = "retained"} @Int
    %cycle = firrtl.object {test.attr = "retained"} @Int(in input: !firrtl.integer, out output: !firrtl.integer)
    // CHECK-NEXT: %[[IN:.*]] = firrtl.object.subfield %cycle[input]
    %cycle_in = firrtl.object.subfield %cycle[input] : !firrtl.class<@Int(in input: !firrtl.integer, out output: !firrtl.integer)>
    // CHECK-NEXT: %[[OUT:.*]] = firrtl.object.subfield %cycle[output]
    %cycle_out = firrtl.object.subfield %cycle[output] : !firrtl.class<@Int(in input: !firrtl.integer, out output: !firrtl.integer)>
    // CHECK-NEXT: %[[C:.*]] = firrtl.integer 1
    %constant = firrtl.integer 1
    // CHECK-NEXT: %[[SUM:.*]] = firrtl.integer.add %[[C]], %[[OUT]]
    %sum = firrtl.integer.add %constant, %cycle_out : (!firrtl.integer, !firrtl.integer) -> !firrtl.integer
    // CHECK-NEXT: firrtl.propassign %[[IN]], %[[SUM]]
    firrtl.propassign %cycle_in, %sum : !firrtl.integer
    // CHECK-NEXT: %independent = firrtl.object.create(%[[C]])
    %independent = firrtl.object @Int(in input: !firrtl.integer, out output: !firrtl.integer)
    %independent_in = firrtl.object.subfield %independent[input] : !firrtl.class<@Int(in input: !firrtl.integer, out output: !firrtl.integer)>
    firrtl.propassign %independent_in, %constant : !firrtl.integer
    // CHECK-NEXT: %consumer = firrtl.object.create(%[[OUT]])
    // CHECK-NEXT: }
    %consumer = firrtl.object @Int(in input: !firrtl.integer, out output: !firrtl.integer)
    %consumer_in = firrtl.object.subfield %consumer[input] : !firrtl.class<@Int(in input: !firrtl.integer, out output: !firrtl.integer)>
    firrtl.propassign %consumer_in, %cycle_out : !firrtl.integer
    // OM: om.object @Int
    // OM: om.object @Int
    // OM: om.object @Int
  }

  firrtl.extmodule @Producer(out prop: !firrtl.integer)

  // Module instances are declarative producers too. Moving an instance before
  // an object also makes the object's earlier output uses valid SSA.
  // CHECK-LABEL: firrtl.module @ForwardInstance
  // OM-LABEL: om.class @ForwardInstance_Class
  firrtl.module @ForwardInstance(out %out: !firrtl.integer) {
    // CHECK-NEXT: %[[PROP:.*]] = firrtl.instance producer @Producer
    // CHECK-NEXT: %obj = firrtl.object.create(%[[PROP]])
    %obj = firrtl.object @Int(in input: !firrtl.integer, out output: !firrtl.integer)
    %in = firrtl.object.subfield %obj[input] : !firrtl.class<@Int(in input: !firrtl.integer, out output: !firrtl.integer)>
    // CHECK-NEXT: %[[OUT:.*]] = firrtl.object.subfield %obj[output]
    %result = firrtl.object.subfield %obj[output] : !firrtl.class<@Int(in input: !firrtl.integer, out output: !firrtl.integer)>
    // CHECK-NEXT: firrtl.propassign %out, %[[OUT]]
    firrtl.propassign %out, %result : !firrtl.integer
    // CHECK-NEXT: }
    %prop = firrtl.instance producer @Producer(out prop: !firrtl.integer)
    firrtl.propassign %in, %prop : !firrtl.integer
    // OM: om.object @Producer_Class
    // OM: om.object @Int
  }

  // FIRRTL producers need not be pure: an existing property wire can be
  // hoisted to make a forward input available, without adding a temporary wire.
  // CHECK-LABEL: firrtl.module @ForwardWire
  // OM-LABEL: om.class @ForwardWire_Class
  firrtl.module @ForwardWire() {
    // CHECK-NEXT: %wire = firrtl.wire : !firrtl.integer
    // CHECK-NEXT: %obj = firrtl.object.create(%wire)
    %obj = firrtl.object @Int(in input: !firrtl.integer, out output: !firrtl.integer)
    %in = firrtl.object.subfield %obj[input] : !firrtl.class<@Int(in input: !firrtl.integer, out output: !firrtl.integer)>
    %wire = firrtl.wire : !firrtl.integer
    // CHECK-NEXT: %[[C:.*]] = firrtl.integer 2
    %constant = firrtl.integer 2
    // CHECK-NEXT: firrtl.propassign %wire, %[[C]]
    firrtl.propassign %wire, %constant : !firrtl.integer
    firrtl.propassign %in, %wire : !firrtl.integer
    // CHECK-NEXT: %independent = firrtl.object.create(%wire)
    // CHECK-NEXT: }
    %independent = firrtl.object @Int(in input: !firrtl.integer, out output: !firrtl.integer)
    %independent_in = firrtl.object.subfield %independent[input] : !firrtl.class<@Int(in input: !firrtl.integer, out output: !firrtl.integer)>
    firrtl.propassign %independent_in, %wire : !firrtl.integer
    // OM: om.object @Int
    // OM: om.object @Int
  }
}
