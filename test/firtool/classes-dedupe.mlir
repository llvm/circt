// RUN: firtool --ir-fir %s | FileCheck %s --check-prefixes=CHECK,CHECK-ON
// RUN: firtool --ir-fir --dedup-classes=0 %s | FileCheck %s --check-prefixes=CHECK,CHECK-OFF
// RUN: firtool --ir-fir --dedup-classes=1 %s | FileCheck %s --check-prefixes=CHECK,CHECK-ON

// Check that the `--dedup-classes` flag actually does something.

// CHECK-LABEL: "ClassDedup"
firrtl.circuit "ClassDedup" {
  // CHECK: firrtl.class private @Foo()
  // CHECK-OFF: firrtl.class private @Bar()
  // CHECK-ON-NOT: firrtl.class private @Bar()
  firrtl.class private @Foo() {}
  firrtl.class private @Bar() {}

  // CHECK: @ClassDedup()
  firrtl.module @ClassDedup() {
    // CHECK-NEXT: firrtl.object @Foo()
    // CHECK-OFF-NEXT: firrtl.object @Bar()
    // CHECK-ON-NEXT: firrtl.object @Foo()
    %obj1 = firrtl.object @Foo()
    %obj2 = firrtl.object @Bar()
  }
}
