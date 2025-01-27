// RUN: circt-opt --strip-om --allow-unregistered-dialect %s | FileCheck %s

// CHECK-NOT: om.class
om.class @Foo() {
  om.class.fields
}

// CHECK-NOT: om.class.extern
om.class.extern @Bar() {}

// CHECK: "some_unknown_dialect.op"
"some_unknown_dialect.op"() {} : () -> ()
