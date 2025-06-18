// RUN: ! (arcilator %s --run --jit-entry=main --until-after=state-alloc 2> %t) && FileCheck --input-file=%t %s
// RUN: ! (arcilator %s --run --jit-entry=main --until-before=state-alloc 2> %t) && FileCheck --input-file=%t %s
// REQUIRES: arcilator-jit

// CHECK: full pipeline must be run for JIT execution

func.func @main(%a: i32) {
  return
}
