// RUN: circt-opt -cse %s | FileCheck %s

// CSE should not result in the deletion of `hw.donttouch`.
//
// CHECK-LABEL: hw.donttouch <@Foo::@a>
hw.donttouch <@Foo::@a>
hw.module public @Foo() {
  %a = sv.wire sym @a : !hw.inout<i1>
}
