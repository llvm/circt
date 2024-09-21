// RUN: circt-opt %s -test-firrtl-instance-info 2>&1 | FileCheck %s

firrtl.circuit "Foo" {
  firrtl.layer @A bind {
  }
  // CHECK:      @Corge
  // CHECK-NEXT:   isDut: false
  // CHECK-NEXT:   isUnderDut: false
  // CHECK-NEXT:   isFullyUnderDut: false
  // CHECK-NEXT:   isUnderLayer: true
  // CHECK-NEXT:   isFullyUnderLayer: false
  firrtl.module private @Corge() {}
  // CHECK:      @Quz
  // CHECK-NEXT:   isDut: false
  // CHECK-NEXT:   isUnderDut: false
  // CHECK-NEXT:   isFullyUnderDut: false
  // CHECK-NEXT:   isUnderLayer: true
  // CHECK-NEXT:   isFullyUnderLayer: true
  firrtl.module private @Quz() {}
  // CHECK:      @Qux
  // CHECK-NEXT:   isDut: false
  // CHECK-NEXT:   isUnderDut: true
  // CHECK-NEXT:   isFullyUnderDut: false
  // CHECK-NEXT:   isUnderLayer: false
  // CHECK-NEXT:   isFullyUnderLayer: false
  firrtl.module private @Qux() {}
  // CHECK:      @Baz
  // CHECK-NEXT:   isDut: false
  // CHECK-NEXT:   isUnderDut: true
  // CHECK-NEXT:   isFullyUnderDut: true
  // CHECK-NEXT:   isUnderLayer: false
  // CHECK-NEXT:   isFullyUnderLayer: false
  firrtl.module private @Baz() {}
  // CHECK:      @Bar
  // CHECK-NEXT:   isDut: true
  // CHECK-NEXT:   isUnderDut: false
  // CHECK-NEXT:   isFullyUnderDut: false
  // CHECK-NEXT:   isUnderLayer: false
  // CHECK-NEXT:   isFullyUnderLayer: false
  firrtl.module private @Bar() attributes {
    annotations = [
      {class = "sifive.enterprise.firrtl.MarkDUTAnnotation"}
    ]
  } {
    firrtl.instance baz @Baz()
    firrtl.instance qux @Qux()
  }
  // CHECK:      @Foo
  // CHECK-NEXT:   isDut: false
  // CHECK-NEXT:   isUnderDut: false
  // CHECK-NEXT:   isFullyUnderDut: false
  // CHECK-NEXT:   isUnderLayer: false
  // CHECK-NEXT:   isFullyUnderLayer: false
  firrtl.module @Foo() {
    firrtl.instance bar @Bar()
    firrtl.instance qux @Qux()
    firrtl.layerblock @A {
      firrtl.instance quz @Quz()
      firrtl.instance corge @Corge()
    }
    firrtl.instance corge2 @Corge()
  }
}
