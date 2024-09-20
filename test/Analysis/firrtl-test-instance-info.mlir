// RUN: circt-opt %s -test-firrtl-instance-info 2>&1 | FileCheck %s

firrtl.circuit "Foo" {
  firrtl.layer @A bind {
  }
  // CHECK:      firrtl.module @Corge
  // CHECK-NEXT:   isDut: false
  // CHECK-NEXT:   isUnderDut: false
  // CHECK-NEXT:   isFullyUnderDut: false
  // CHECK-NEXT:   isUnderLayer: true
  // CHECK-NEXT:   isFullyUnderLayer: false
  firrtl.module @Corge() {}
  // CHECK:      firrtl.module @Quz
  // CHECK-NEXT:   isDut: false
  // CHECK-NEXT:   isUnderDut: false
  // CHECK-NEXT:   isFullyUnderDut: false
  // CHECK-NEXT:   isUnderLayer: true
  // CHECK-NEXT:   isFullyUnderLayer: true
  firrtl.module @Quz() {}
  // CHECK:      firrtl.module @Qux
  // CHECK-NEXT:   isDut: false
  // CHECK-NEXT:   isUnderDut: true
  // CHECK-NEXT:   isFullyUnderDut: false
  // CHECK-NEXT:   isUnderLayer: false
  // CHECK-NEXT:   isFullyUnderLayer: false
  firrtl.module @Qux() {}
  // CHECK:      firrtl.module @Baz
  // CHECK-NEXT:   isDut: false
  // CHECK-NEXT:   isUnderDut: true
  // CHECK-NEXT:   isFullyUnderDut: true
  // CHECK-NEXT:   isUnderLayer: false
  // CHECK-NEXT:   isFullyUnderLayer: false
  firrtl.module @Baz() {}
  // CHECK:      firrtl.module @Bar
  // CHECK-NEXT:   isDut: true
  // CHECK-NEXT:   isUnderDut: false
  // CHECK-NEXT:   isFullyUnderDut: false
  // CHECK-NEXT:   isUnderLayer: false
  // CHECK-NEXT:   isFullyUnderLayer: false
  firrtl.module @Bar() attributes {annotations = [{class = "sifive.enterprise.firrtl.MarkDUTAnnotation"}]} {
    firrtl.instance baz interesting_name @Baz()
    firrtl.instance qux interesting_name @Qux()
  }
  // CHECK: firrtl.module @Foo
  // CHECK-NEXT:   isDut: false
  // CHECK-NEXT:   isUnderDut: false
  // CHECK-NEXT:   isFullyUnderDut: false
  // CHECK-NEXT:   isUnderLayer: false
  // CHECK-NEXT:   isFullyUnderLayer: false
  firrtl.module @Foo() {
    firrtl.instance bar interesting_name @Bar()
    firrtl.instance qux interesting_name @Qux()
    firrtl.layerblock @A {
      firrtl.instance quz interesting_name @Quz()
      firrtl.instance corge interesting_name @Corge()
    }
    firrtl.instance corge2 interesting_name @Corge()
  }
}
