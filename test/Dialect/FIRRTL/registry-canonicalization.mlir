// RUN: circt-opt --canonicalize --cse %s | FileCheck %s

firrtl.circuit "Registries" {
  // Unused registries can be removed, but distinct registries must not merge.
  // CHECK-LABEL: firrtl.module @Registries
  // CHECK: %[[ONE:.*]] = firrtl.integer 1
  // CHECK-NEXT: %[[FIRST:.*]] = firrtl.prop.registry : !firrtl.registry<integer>
  // CHECK-NEXT: %[[SECOND:.*]] = firrtl.prop.registry : !firrtl.registry<integer>
  // CHECK-NOT: firrtl.prop.registry
  // CHECK: firrtl.prop.insert %[[FIRST]], %[[ONE]]
  // CHECK-NEXT: firrtl.prop.insert %[[FIRST]], %[[ONE]]
  // CHECK: firrtl.propassign %a, %[[FIRST]]
  // CHECK: firrtl.propassign %b, %[[SECOND]]
  firrtl.module @Registries(out %a: !firrtl.registry<integer>, out %b: !firrtl.registry<integer>) {
    %first = firrtl.prop.registry : !firrtl.registry<integer>
    %second = firrtl.prop.registry : !firrtl.registry<integer>
    %unused = firrtl.prop.registry : !firrtl.registry<integer>
    %one = firrtl.integer 1
    firrtl.prop.insert %first, %one : !firrtl.registry<integer>, !firrtl.integer
    firrtl.prop.insert %first, %one : !firrtl.registry<integer>, !firrtl.integer
    firrtl.propassign %a, %first : !firrtl.registry<integer>
    firrtl.propassign %b, %second : !firrtl.registry<integer>
  }
}
