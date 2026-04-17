// RUN: firtool --help | FileCheck %s

// CHECK: OVERVIEW: MLIR-based FIRRTL compiler
// CHECK-DAG: --lower-to-core
// CHECK-DAG: General {{[Oo]}}ptions:
// CHECK-DAG: Generic Options:
// CHECK-DAG: firtool Options:
// CHECK-DAG: -j{{.*}}Alias for --num-threads
// CHECK-DAG: --lowering-options=
// CHECK-DAG: --num-threads=<N>{{.*}}Number of threads to use for parallel compilation
