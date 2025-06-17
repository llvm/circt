// RUN: not circt-translate --import-aiger %S/delta-underflow.aig --split-input-file 2>&1 | FileCheck %s
// CHECK: invalid binary AND gate: delta causes underflow
