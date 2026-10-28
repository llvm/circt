// RUN: circt-opt --firrtl-dedup %s | FileCheck %s

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

// Modules using different nominal classes must remain distinct, while modules
// using the same class may still be deduplicated.
// CHECK-LABEL: "ModulesWithClasses"
firrtl.circuit "ModulesWithClasses" {
  // CHECK: firrtl.class private @Foo()
  // CHECK: firrtl.class private @Bar()
  firrtl.class private @Foo() {}
  firrtl.class private @Bar() {}

  // CHECK: firrtl.module private @UseFoo()
  firrtl.module private @UseFoo() {
    %obj = firrtl.object @Foo()
  }
  // CHECK: firrtl.module private @UseBar()
  firrtl.module private @UseBar() {
    %obj = firrtl.object @Bar()
  }
  // CHECK-NOT: firrtl.module private @UseFooAgain()
  firrtl.module private @UseFooAgain() {
    %obj = firrtl.object @Foo()
  }

  // CHECK: firrtl.module @ModulesWithClasses()
  firrtl.module @ModulesWithClasses() {
    // CHECK-NEXT: firrtl.instance foo @UseFoo()
    // CHECK-NEXT: firrtl.instance bar @UseBar()
    // CHECK-NEXT: firrtl.instance fooAgain @UseFoo()
    firrtl.instance foo @UseFoo()
    firrtl.instance bar @UseBar()
    firrtl.instance fooAgain @UseFooAgain()
  }
}
