// RUN: circt-translate --import-aiger %S/invalid-index.aig | FileCheck %s
// This AIGER file has leading whitespace in AND gate definition which
// a parser incorrectly skipped.

// CHECK: @aiger_top
