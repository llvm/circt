// UNSUPPORTED: system-windows
//   See https://github.com/llvm/circt/issues/4129
// RUN: circt-reduce %s --test /usr/bin/env --test-arg true --include make-symbols-private | FileCheck %s

// This test verifies that the symbol visibility reducer changes public symbols to private

// CHECK-LABEL: func.func private @publicFunc
func.func public @publicFunc() {
  return
}

// CHECK-LABEL: func.func private @anotherPublicFunc
func.func public @anotherPublicFunc() {
  return
}

// This should remain unchanged as it's already private
// CHECK-LABEL: func.func private @privateFunc
func.func private @privateFunc() {
  return
}

// This should be changed from default (public) to explicit private
// CHECK-LABEL: func.func private @defaultFunc
func.func @defaultFunc() {
  return
}
