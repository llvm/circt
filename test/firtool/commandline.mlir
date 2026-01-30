// RUN: firtool --help | FileCheck %s --implicit-check-not='{{[Oo]}}ptions:'
// RUN: firtool --help-hidden | FileCheck %s --check-prefix=HIDDEN

// CHECK: OVERVIEW: MLIR-based FIRRTL compiler
// CHECK: General {{[Oo]}}ptions
// CHECK: Generic Options
// CHECK: firtool Options
// CHECK-DAG: -j{{.*}}Alias for --num-threads
// CHECK-DAG: --lowering-options=
// CHECK-DAG: --num-threads=<N>{{.*}}Number of threads to use for parallel compilation

// HIDDEN: --verify=<value>
// HIDDEN-SAME: Specify when to run verification
// HIDDEN-DAG: =all
// HIDDEN-DAG: =default
// HIDDEN-DAG: =none
