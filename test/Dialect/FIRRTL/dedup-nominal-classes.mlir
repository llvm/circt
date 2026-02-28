// RUN: circt-opt --firrtl-dedup=dedup-classes=0 %s | FileCheck %s

// CHECK-LABEL: "DontDedupClasses"
firrtl.circuit "DontDedupClasses" {
  // CHECK: firrtl.class private @Foo()
  // CHECK: firrtl.class private @Bar()
  firrtl.class private @Foo() {}
  firrtl.class private @Bar() {}

  // CHECK: @DontDedupClasses()
  firrtl.module @DontDedupClasses() {
    // CHECK-NEXT: firrtl.wire : !firrtl.class<@Foo()>
    // CHECK-NEXT: firrtl.wire : !firrtl.class<@Bar()>
    %wire1 = firrtl.wire : !firrtl.class<@Foo()>
    %wire2 = firrtl.wire : !firrtl.class<@Bar()>
  }
}
