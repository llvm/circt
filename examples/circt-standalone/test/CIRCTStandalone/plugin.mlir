// UNSUPPORTED: system-windows
// UNSUPPORTED: no-circt-standalone-plugin
// RUN: circt-opt %s --load-dialect-plugin=%circt_standalone_libs/CIRCTStandalonePlugin%shlibext --pass-pipeline="builtin.module(circt-standalone-rename-hw-module)" | FileCheck %s

// CHECK-LABEL: hw.module @foo()
hw.module @bar() {
}
