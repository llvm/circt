// RUN: ! (arcilator %s --run --jit-entry=main 2> %t) && FileCheck --input-file=%t %s
// REQUIRES: arcilator-jit

// CHECK: entry point 'main' must have no arguments

func.func @main(%a: i32) {
  return
}
