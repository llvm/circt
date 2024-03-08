// RUN: ! (arcilator %s --run=main 2> %t) && (cat %t | FileCheck %s)

// CHECK: entry point 'main' must have no arguments

func.func @main(%a: i32) {
    return
}
