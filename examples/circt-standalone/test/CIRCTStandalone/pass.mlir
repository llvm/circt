// RUN: circt-standalone-opt %s --pass-pipeline="builtin.module(circt-standalone-rename-hw-module)" | FileCheck %s

// CHECK-LABEL: hw.module @foo()
hw.module @bar() {
}
