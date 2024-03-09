// RUN: ! (arcilator %s --run=unknown 2> %t) && (cat %t | FileCheck %s)
// REQUIRES: arcilator-jit

// CHECK: entry point not found: 'unknown'

func.func @main() {
    return
}
