// RUN: env Z3LIB=%libz3 not circt-test %S/basic.mlir -r \circt-bmc 2>&1 | FileCheck %S/basic.mlir
// REQUIRES: libz3
// REQUIRES: circt-bmc-jit
