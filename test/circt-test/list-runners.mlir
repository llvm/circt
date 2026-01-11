// RUN: circt-test --list-runners | FileCheck %s

// CHECK-DAG: sby formal {{.*}}circt-test-runner-sby.py
// CHECK-DAG: circt-bmc formal {{.*}}circt-test-runner-circt-bmc.py
