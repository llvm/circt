// RUN: circt-opt %s -pass-pipeline='builtin.module(firrtl.circuit(firrtl.module(firrtl-promote-objects)))' | FileCheck %s

firrtl.circuit "InputRead" {
  firrtl.extclass @Input(in value: !firrtl.string)
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
}

firrtl.circuit "DuplicateDrivers" {
  firrtl.extclass @Input(in value: !firrtl.string)
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
}

// A failed dependency must not partially promote either object, but unrelated
// objects should still be promoted.
firrtl.circuit "FailedDependency" {
  firrtl.extclass @Source(in value: !firrtl.string, out result: !firrtl.string)
  firrtl.extclass @Sink(in value: !firrtl.string)
  firrtl.extclass @Empty()
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

firrtl.circuit "MissingDriver" {
  firrtl.extclass @Input(in value: !firrtl.string)
  firrtl.extclass @Empty()
  // CHECK-LABEL: firrtl.module @MissingDriver
  firrtl.module @MissingDriver() {
    // CHECK: firrtl.object @Input
    // CHECK: firrtl.object.create() : () -> !firrtl.class<@Empty()>
    %obj = firrtl.object @Input(in value: !firrtl.string)
    %other = firrtl.object @Empty()
  }
}
