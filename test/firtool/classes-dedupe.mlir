// RUN: firtool --ir-fir %s | FileCheck %s
// RUN: firtool --ir-fir --dedup-classes=1 %s 2>&1 | FileCheck %s --check-prefixes=CHECK,WARN

// WARN: warning: --dedup-classes is deprecated and ignored; classes are never deduplicated

// Classes are nominally typed and should not be deduplicated by firtool.

// CHECK-LABEL: "ClassDedup"
firrtl.circuit "ClassDedup" {
  // CHECK: firrtl.class private @Foo()
  // CHECK: firrtl.class private @Bar()
  firrtl.class private @Foo() {}
  firrtl.class private @Bar() {}

  // CHECK: @ClassDedup()
  firrtl.module @ClassDedup() {
    // CHECK-NEXT: %obj1 = firrtl.object.create() : () -> !firrtl.class<@Foo()>
    // CHECK-NEXT: %obj2 = firrtl.object.create() : () -> !firrtl.class<@Bar()>
    %obj1 = firrtl.object @Foo()
    %obj2 = firrtl.object @Bar()
  }
}
