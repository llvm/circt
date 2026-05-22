// RUN: not om-linker %S/Inputs/elaborate-def.mlir %S/Inputs/elaborate-use.mlir 2>&1 | FileCheck %s

// CHECK: error: OM property assertion failed: linked child condition must hold
