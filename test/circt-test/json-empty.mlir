// RUN: circt-test -l %s 2>&1 | FileCheck %s --check-prefix=NO-JSON
// RUN: circt-test -l --json %s | FileCheck %s --check-prefix=EMPTY-JSON

// NO-JSON: no tests discovered
// EMPTY-JSON: []

builtin.module {}
