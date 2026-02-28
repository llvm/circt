// RUN: circt-opt --strip-emit --allow-unregistered-dialect %s | FileCheck %s

// CHECK-NOT: emit.file
emit.file "foo.sv" {}

// CHECK-NOT: emit.fragment
emit.fragment @Bar {}

// CHECK-NOT: emit.file_list
emit.file_list "baz.f", []

// CHECK: "some_unknown_dialect.op"
"some_unknown_dialect.op"() {} : () -> ()

// CHECK: hw.module @Baz
// CHECK-NOT: emit.fragments
hw.module @Baz() attributes {emit.fragments = [@Bar]} {
  hw.output
}
