// RUN: firtool --help | FileCheck %s --implicit-check-not='{{[Oo]}}ptions:'

// CHECK: OVERVIEW: MLIR-based FIRRTL compiler
// CHECK: General {{[Oo]}}ptions
// CHECK: Generic Options
// CHECK: firtool Options
// CHECK-DAG: -j{{.*}}Alias for --num-threads
// CHECK-DAG: --lowering-options=
// CHECK-DAG: --num-threads=<N>{{.*}}Number of threads to use for parallel compilation
