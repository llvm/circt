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

  // CHECK: @ClassDedup(
  firrtl.module @ClassDedup(out %a: !firrtl.anyref, out %b: !firrtl.anyref) {
    // CHECK-NEXT: firrtl.object.create() : () -> !firrtl.class<@Foo()>
    // CHECK-OFF-NEXT: firrtl.object.create() : () -> !firrtl.class<@Bar()>
    // CHECK-ON-NEXT: firrtl.object.create() : () -> !firrtl.class<@Foo()>
    %obj1 = firrtl.object @Foo()
    %obj2 = firrtl.object @Bar()
    // Keep the value-like objects live through optimization.
    %ref1 = firrtl.object.anyref_cast %obj1 : !firrtl.class<@Foo()>
    %ref2 = firrtl.object.anyref_cast %obj2 : !firrtl.class<@Bar()>
    firrtl.propassign %a, %ref1 : !firrtl.anyref
    firrtl.propassign %b, %ref2 : !firrtl.anyref
  }
}
