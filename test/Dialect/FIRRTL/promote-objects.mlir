// RUN: circt-opt %s -pass-pipeline='builtin.module(firrtl.circuit(firrtl.module(firrtl-promote-objects)))' | FileCheck %s

firrtl.circuit "Objects" {
  firrtl.class @Pair(in %first: !firrtl.string, out %result: !firrtl.string, in %second: !firrtl.integer) {
    firrtl.propassign %result, %first : !firrtl.string
  }
  firrtl.extclass @Empty()
  firrtl.extclass @Source(in value: !firrtl.string, out result: !firrtl.string)
  firrtl.extclass @Sink(in value: !firrtl.string)
  firrtl.extclass @Input(in value: !firrtl.string)

  // CHECK-LABEL: firrtl.module @Objects
  firrtl.module @Objects(in %arg: !firrtl.string, out %out: !firrtl.string, out %ref: !firrtl.anyref) {
    // CHECK-NEXT: %[[I:.*]] = firrtl.integer 42
    // CHECK-NEXT: %pair = firrtl.object.create(%arg, %[[I]]) : (!firrtl.string, !firrtl.integer) -> !firrtl.class<@Pair(in first: !firrtl.string, out result: !firrtl.string, in second: !firrtl.integer)>
    %pair = firrtl.object @Pair(in first: !firrtl.string, out result: !firrtl.string, in second: !firrtl.integer)
    %first = firrtl.object.subfield %pair[first] : !firrtl.class<@Pair(in first: !firrtl.string, out result: !firrtl.string, in second: !firrtl.integer)>
    %duplicate = firrtl.object.subfield %pair[first] : !firrtl.class<@Pair(in first: !firrtl.string, out result: !firrtl.string, in second: !firrtl.integer)>
    // CHECK-NEXT: %[[RESULT:.*]] = firrtl.object.subfield %pair[result]
    %result = firrtl.object.subfield %pair[result] : !firrtl.class<@Pair(in first: !firrtl.string, out result: !firrtl.string, in second: !firrtl.integer)>
    %second = firrtl.object.subfield %pair[second] : !firrtl.class<@Pair(in first: !firrtl.string, out result: !firrtl.string, in second: !firrtl.integer)>
    %int = firrtl.integer 42
    firrtl.propassign %second, %int : !firrtl.integer
    firrtl.propassign %first, %arg : !firrtl.string
    // CHECK-NEXT: firrtl.propassign %out, %[[RESULT]] : !firrtl.string
    firrtl.propassign %out, %result : !firrtl.string
    // CHECK-NEXT: %[[REF:.*]] = firrtl.object.anyref_cast %pair
    %cast = firrtl.object.anyref_cast %pair : !firrtl.class<@Pair(in first: !firrtl.string, out result: !firrtl.string, in second: !firrtl.integer)>
    // CHECK-NEXT: firrtl.propassign %ref, %[[REF]] : !firrtl.anyref
    firrtl.propassign %ref, %cast : !firrtl.anyref
    // CHECK-NEXT: %empty = firrtl.object.create() : () -> !firrtl.class<@Empty()>
    // CHECK-NEXT: }
    %empty = firrtl.object @Empty()
  }

  // A later object and its output must be hoisted to feed the first object.
  // The third object then uses the hoisted operations' shared position.
  // CHECK-LABEL: firrtl.module @Dependencies
  firrtl.module @Dependencies(in %arg: !firrtl.string) {
    // CHECK: %[[SRC:.*]] = firrtl.object.create(%arg)
    // CHECK-NEXT: %[[RESULT:.*]] = firrtl.object.subfield %[[SRC]][result]
    // CHECK-NEXT: firrtl.object.create(%[[RESULT]])
    // CHECK-NEXT: firrtl.object.create(%[[RESULT]])
    %first = firrtl.object @Sink(in value: !firrtl.string)
    %src = firrtl.object @Source(in value: !firrtl.string, out result: !firrtl.string)
    %third = firrtl.object @Sink(in value: !firrtl.string)
    %input = firrtl.object.subfield %src[value] : !firrtl.class<@Source(in value: !firrtl.string, out result: !firrtl.string)>
    %result = firrtl.object.subfield %src[result] : !firrtl.class<@Source(in value: !firrtl.string, out result: !firrtl.string)>
    %firstInput = firrtl.object.subfield %first[value] : !firrtl.class<@Sink(in value: !firrtl.string)>
    %thirdInput = firrtl.object.subfield %third[value] : !firrtl.class<@Sink(in value: !firrtl.string)>
    firrtl.propassign %input, %arg : !firrtl.string
    firrtl.propassign %firstInput, %result : !firrtl.string
    firrtl.propassign %thirdInput, %result : !firrtl.string
  }

  // CHECK-LABEL: firrtl.module @InputRead
  firrtl.module @InputRead(in %arg: !firrtl.string) {
    // CHECK: %[[OBJ:.*]] = firrtl.object @Input
    // CHECK: %[[FIELD:.*]] = firrtl.object.subfield %[[OBJ]][value]
    // CHECK: firrtl.string.concat %[[FIELD]], %arg
    // CHECK: firrtl.propassign %[[FIELD]], %arg
    %obj = firrtl.object @Input(in value: !firrtl.string)
    %field = firrtl.object.subfield %obj[value] : !firrtl.class<@Input(in value: !firrtl.string)>
    %read = firrtl.string.concat %field, %arg : !firrtl.string
    firrtl.propassign %field, %arg : !firrtl.string
  }

  // CHECK-LABEL: firrtl.module @DuplicateDrivers
  firrtl.module @DuplicateDrivers(in %arg: !firrtl.string) {
    // CHECK: %[[OBJ:.*]] = firrtl.object @Input
    // CHECK: %[[FIRST:.*]] = firrtl.object.subfield %[[OBJ]][value]
    // CHECK: %[[SECOND:.*]] = firrtl.object.subfield %[[OBJ]][value]
    // CHECK: firrtl.propassign %[[FIRST]], %arg
    // CHECK: firrtl.propassign %[[SECOND]], %arg
    %obj = firrtl.object @Input(in value: !firrtl.string)
    %first = firrtl.object.subfield %obj[value] : !firrtl.class<@Input(in value: !firrtl.string)>
    %second = firrtl.object.subfield %obj[value] : !firrtl.class<@Input(in value: !firrtl.string)>
    firrtl.propassign %first, %arg : !firrtl.string
    firrtl.propassign %second, %arg : !firrtl.string
  }

  // A failed dependency must not partially promote either object, but unrelated
  // objects should still be promoted.
  // CHECK-LABEL: firrtl.module @FailedDependency
  firrtl.module @FailedDependency() {
    // CHECK: %[[SINK:.*]] = firrtl.object @Sink
    // CHECK: %[[SRC:.*]] = firrtl.object @Source
    // CHECK: firrtl.object.subfield %[[SRC]][result]
    // CHECK: firrtl.object.subfield %[[SINK]][value]
    // CHECK: firrtl.propassign
    // CHECK: firrtl.object.create() : () -> !firrtl.class<@Empty()>
    %sink = firrtl.object @Sink(in value: !firrtl.string)
    %src = firrtl.object @Source(in value: !firrtl.string, out result: !firrtl.string)
    %result = firrtl.object.subfield %src[result] : !firrtl.class<@Source(in value: !firrtl.string, out result: !firrtl.string)>
    %value = firrtl.object.subfield %sink[value] : !firrtl.class<@Sink(in value: !firrtl.string)>
    firrtl.propassign %value, %result : !firrtl.string
    %other = firrtl.object @Empty()
  }
}
