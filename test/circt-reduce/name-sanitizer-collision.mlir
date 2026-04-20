// UNSUPPORTED: system-windows
// RUN: circt-reduce %s --include=module-name-sanitizer --test /usr/bin/env --test-arg true --keep-best=0 | FileCheck %s

// Test that a metasyntactic name collision doesn't result in a symbol collision
// error.  Run this test separately to avoid collisions with other circuits as
// we don't have a `-split-input-file` option to `circt-reduce`.

// CHECK-LABEL: firrtl.circuit "Foo_0"
// CHECK:       firrtl.module @Foo_0()
// CHECK:       firrtl.extmodule @Qux()
// CHECK:       firrtl.extmodule @Foo()
firrtl.circuit "A" {
  firrtl.module @A() {
  }
  firrtl.extmodule @Qux()
  firrtl.extmodule @Foo()
}
