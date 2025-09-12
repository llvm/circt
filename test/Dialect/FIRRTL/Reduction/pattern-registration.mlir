// UNSUPPORTED: system-windows
//   See https://github.com/llvm/circt/issues/4129

// This test checks that only the reduction patterns of dialects that occur in
// the input file are registered

// RUN: circt-reduce %s --test /usr/bin/env --test-arg cat --list | FileCheck %s

// CHECK:     symbol-dce
// CHECK-DAG: annotation-remover
// CHECK-DAG: canonicalize
// CHECK-DAG: connect-forwarder
// CHECK-DAG: connect-invalidator
// CHECK-DAG: connect-source-operand-0-forwarder
// CHECK-DAG: connect-source-operand-1-forwarder
// CHECK-DAG: connect-source-operand-2-forwarder
// CHECK-DAG: cse
// CHECK-DAG: detach-subaccesses
// CHECK-DAG: eager-inliner
// CHECK-DAG: extmodule-instance-remover
// CHECK-DAG: firrtl-constantifier
// CHECK-DAG: firrtl-expand-whens
// CHECK-DAG: firrtl-force-dedup
// CHECK-DAG: firrtl-imconstprop
// CHECK-DAG: firrtl-infer-resets
// CHECK-DAG: firrtl-infer-widths
// CHECK-DAG: firrtl-inliner
// CHECK-DAG: firrtl-lower-chirrtl
// CHECK-DAG: firrtl-lower-types
// CHECK-DAG: firrtl-module-externalizer
// CHECK-DAG: firrtl-module-swapper
// CHECK-DAG: firrtl-operand0-forwarder
// CHECK-DAG: firrtl-operand1-forwarder
// CHECK-DAG: firrtl-operand2-forwarder
// CHECK-DAG: firrtl-remove-unused-ports
// CHECK-DAG: hw-constantifier
// CHECK-DAG: hw-module-externalizer
// CHECK-DAG: hw-operand0-forwarder
// CHECK-DAG: hw-operand1-forwarder
// CHECK-DAG: hw-operand2-forwarder
// CHECK-DAG: instance-stubber
// CHECK-DAG: memory-stubber
// CHECK-DAG: module-internal-name-sanitizer
// CHECK-DAG: module-name-sanitizer
// CHECK-DAG: node-symbol-remover
// CHECK-DAG: operation-pruner
// CHECK-DAG: root-port-pruner
// CHECK-EMPTY:
firrtl.circuit "Foo" {
  firrtl.module @Foo() {}
}
