// UNSUPPORTED: system-windows
//   See https://github.com/llvm/circt/issues/4129

// This test checks that only the reduction patterns of dialects that occur in
// the input file are registered

// RUN: circt-reduce %s --test /usr/bin/env --test-arg cat --list | FileCheck %s

// CHECK-DAG: hw-module-externalizer
// CHECK-DAG: hw-constantifier
// CHECK-DAG: hw-operand0-forwarder
// CHECK-DAG: cse
// CHECK-DAG: hw-operand1-forwarder
// CHECK-DAG: canonicalize
// CHECK-DAG: hw-operand2-forwarder
// CHECK-DAG: hw-module-output-pruner
// CHECK-DAG: hw-module-input-pruner
// CHECK-DAG: operation-pruner
hw.module @Foo(in %in : i1, out out : i1) {
  hw.output %in : i1
}
