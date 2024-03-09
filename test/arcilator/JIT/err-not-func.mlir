// RUN: ! (arcilator %s --run=main 2> %t) && (cat %t | FileCheck %s)
// REQUIRES: arcilator-jit

// CHECK: entry point 'main' was found but on an operation of type 'hw.module' while a function was expected
// CHECK: supported functions: 'func.func', 'llvm.func'

hw.module @main() {
    hw.output
}
