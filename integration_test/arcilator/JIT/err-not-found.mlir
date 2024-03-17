// RUN: ! (arcilator %s --run --jit-entry=unknown 2> %t) && FileCheck --input-file=%t %s
// REQUIRES: arcilator-jit

// CHECK: entry point not found: 'unknown'

func.func @main() {
  return
}
