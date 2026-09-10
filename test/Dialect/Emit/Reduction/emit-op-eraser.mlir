// UNSUPPORTED: system-windows
//   See https://github.com/llvm/circt/issues/4129
// RUN: circt-reduce %s --test /usr/bin/env --test-arg true --include emit-op-eraser --keep-best=0 | FileCheck %s

// CHECK-NOT: emit.file
emit.file "foo" {
  // CHECK-NOT: emit.verbatim
  emit.verbatim "bar"
}
