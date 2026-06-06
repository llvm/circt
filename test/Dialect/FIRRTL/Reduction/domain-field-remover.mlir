// UNSUPPORTED: system-windows
//   See https://github.com/llvm/circt/issues/4129
// RUN: circt-reduce %s --test /usr/bin/env --test-arg true --keep-best=0 --include domain-field-remover | FileCheck %s

// Test removing a single field from a domain
// CHECK-LABEL: firrtl.circuit "Foo"
firrtl.circuit "Foo" {
  // CHECK-NEXT: firrtl.domain @ClockDomain{{$}}
  firrtl.domain @ClockDomain [#firrtl.domain.field<"id", !firrtl.integer>]
  // CHECK-NEXT: firrtl.module @Foo
  // CHECK-SAME: in %A: !firrtl.domain<@ClockDomain()>
  firrtl.module @Foo(in %A: !firrtl.domain<@ClockDomain(id: !firrtl.integer)>) {
    // CHECK-NEXT: %a = firrtl.wire interesting_name : !firrtl.domain<@ClockDomain()>
    %a = firrtl.wire interesting_name : !firrtl.domain<@ClockDomain(id: !firrtl.integer)>
    // CHECK-NEXT: }
  }
}

// Test removing a field that has subfield uses
// CHECK-LABEL: firrtl.circuit "SubfieldTest"
firrtl.circuit "SubfieldTest" {
  // CHECK-NEXT: firrtl.domain @ClockDomain{{$}}
  firrtl.domain @ClockDomain [#firrtl.domain.field<"id", !firrtl.integer>]
  // CHECK-NEXT: firrtl.module @SubfieldTest
  // CHECK-SAME: in %A: !firrtl.domain<@ClockDomain()>
  // CHECK-SAME: out %id: !firrtl.integer
  firrtl.module @SubfieldTest(
    in %A: !firrtl.domain<@ClockDomain(id: !firrtl.integer)>,
    out %id: !firrtl.integer
  ) {
    // CHECK-NEXT: %[[UNKNOWN:.+]] = firrtl.unknown : !firrtl.integer
    // CHECK-NEXT: firrtl.propassign %id, %[[UNKNOWN]] : !firrtl.integer
    %0 = firrtl.domain.subfield %A[id] : !firrtl.domain<@ClockDomain(id: !firrtl.integer)>
    firrtl.propassign %id, %0 : !firrtl.integer
    // CHECK-NEXT: }
  }
}

