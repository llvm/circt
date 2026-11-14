// RUN: circt-opt %s -pass-pipeline='builtin.module(firrtl.circuit(firrtl.module(firrtl-promote-objects)))' | FileCheck %s

!pair = !firrtl.class<@Pair(in first: !firrtl.string, out result: !firrtl.string, in second: !firrtl.integer)>
!source = !firrtl.class<@Source(in value: !firrtl.string, out result: !firrtl.string)>
!sink = !firrtl.class<@Sink(in value: !firrtl.string)>

firrtl.circuit "Objects" {
  firrtl.class @Pair(in %first: !firrtl.string, out %result: !firrtl.string, in %second: !firrtl.integer) {
    firrtl.propassign %result, %first : !firrtl.string
  }
  firrtl.extclass @Source(in value: !firrtl.string, out result: !firrtl.string)
  firrtl.extclass @Sink(in value: !firrtl.string)
  firrtl.extclass @Empty()

  // CHECK-LABEL: firrtl.module @Objects
  firrtl.module @Objects(in %arg: !firrtl.string, out %out: !firrtl.string, out %ref: !firrtl.anyref) {
    // CHECK-NEXT: %[[I:.*]] = firrtl.integer 42
    // CHECK-NEXT: %pair = firrtl.object.create(%arg, %[[I]])
    %pair = firrtl.object @Pair(in first: !firrtl.string, out result: !firrtl.string, in second: !firrtl.integer)
    %first = firrtl.object.subfield %pair[first] : !pair
    // CHECK-NEXT: %[[RESULT:.*]] = firrtl.object.subfield %pair[result]
    %result = firrtl.object.subfield %pair[result] : !pair
    %second = firrtl.object.subfield %pair[second] : !pair
    %int = firrtl.integer 42
    firrtl.propassign %second, %int : !firrtl.integer
    firrtl.propassign %first, %arg : !firrtl.string
    // CHECK-NEXT: firrtl.propassign %out, %[[RESULT]] : !firrtl.string
    firrtl.propassign %out, %result : !firrtl.string
    // CHECK-NEXT: %[[REF:.*]] = firrtl.object.anyref_cast %pair
    %cast = firrtl.object.anyref_cast %pair : !pair
    // CHECK-NEXT: firrtl.propassign %ref, %[[REF]] : !firrtl.anyref
    // CHECK-NEXT: }
    firrtl.propassign %ref, %cast : !firrtl.anyref
  }

  // Hoist a later object and its output, which are also used by a third object.
  // CHECK-LABEL: firrtl.module @Dependencies
  firrtl.module @Dependencies(in %arg: !firrtl.string) {
    // CHECK: %[[SRC:.*]] = firrtl.object.create(%arg)
    // CHECK-NEXT: %[[RESULT:.*]] = firrtl.object.subfield %[[SRC]][result]
    // CHECK-NEXT: firrtl.object.create(%[[RESULT]])
    // CHECK-NEXT: firrtl.object.create(%[[RESULT]])
    %first = firrtl.object @Sink(in value: !firrtl.string)
    %src = firrtl.object @Source(in value: !firrtl.string, out result: !firrtl.string)
    %third = firrtl.object @Sink(in value: !firrtl.string)
    %input = firrtl.object.subfield %src[value] : !source
    %result = firrtl.object.subfield %src[result] : !source
    %firstInput = firrtl.object.subfield %first[value] : !sink
    %thirdInput = firrtl.object.subfield %third[value] : !sink
    firrtl.propassign %input, %arg : !firrtl.string
    firrtl.propassign %firstInput, %result : !firrtl.string
    firrtl.propassign %thirdInput, %result : !firrtl.string
  }

  // CHECK-LABEL: firrtl.module @Unpromoted
  firrtl.module @Unpromoted(in %arg: !firrtl.string) {
    // Reading an input prevents promotion.
    // CHECK: %[[READ:.*]] = firrtl.object @Sink
    // CHECK-NEXT: %[[FIELD:.*]] = firrtl.object.subfield %[[READ]][value]
    // CHECK-NEXT: firrtl.string.concat %[[FIELD]], %arg
    // CHECK-NEXT: firrtl.propassign %[[FIELD]], %arg
    %read = firrtl.object @Sink(in value: !firrtl.string)
    %field = firrtl.object.subfield %read[value] : !sink
    %use = firrtl.string.concat %field, %arg : !firrtl.string
    firrtl.propassign %field, %arg : !firrtl.string

    // Multiple drivers prevent promotion, even through different subfields.
    // CHECK-NEXT: %multi = firrtl.object @Sink
    %multi = firrtl.object @Sink(in value: !firrtl.string)
    %first = firrtl.object.subfield %multi[value] : !sink
    %second = firrtl.object.subfield %multi[value] : !sink
    firrtl.propassign %first, %arg : !firrtl.string
    firrtl.propassign %second, %arg : !firrtl.string

    // A missing driver prevents promotion of both objects, but not unrelated ones.
    // CHECK: %[[SINK:.*]] = firrtl.object @Sink
    // CHECK-NEXT: %[[SRC:.*]] = firrtl.object @Source
    // CHECK-NEXT: %[[RESULT:.*]] = firrtl.object.subfield %[[SRC]][result]
    // CHECK-NEXT: %[[VALUE:.*]] = firrtl.object.subfield %[[SINK]][value]
    // CHECK-NEXT: firrtl.propassign %[[VALUE]], %[[RESULT]]
    // CHECK-NEXT: firrtl.object.create() : () -> !firrtl.class<@Empty()>
    %sink = firrtl.object @Sink(in value: !firrtl.string)
    %src = firrtl.object @Source(in value: !firrtl.string, out result: !firrtl.string)
    %result = firrtl.object.subfield %src[result] : !source
    %value = firrtl.object.subfield %sink[value] : !sink
    firrtl.propassign %value, %result : !firrtl.string
    %other = firrtl.object @Empty()
  }
}
