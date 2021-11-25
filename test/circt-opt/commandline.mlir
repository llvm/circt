// RUN: circt-opt --help | FileCheck %s --check-prefix=HELP
// RUN: circt-opt --show-dialects | FileCheck %s --check-prefix=DIALECT

// HELP: OVERVIEW: CIRCT modular optimizer driver

// DIALECT: Available Dialects:
// DIALECT-NEXT: affine
// DIALECT-NEXT: apint
// DIALECT-NEXT: arith
// DIALECT-NEXT: builtin
// DIALECT-NEXT: calyx
// DIALECT-NEXT: comb
// DIALECT-NEXT: esi
// DIALECT-NEXT: firrtl
// DIALECT-NEXT: fsm
// DIALECT-NEXT: handshake
// DIALECT-NEXT: hw
// DIALECT-NEXT: llhd
// DIALECT-NEXT: llvm
// DIALECT-NEXT: memref
// DIALECT-NEXT: msft
// DIALECT-NEXT: scf
// DIALECT-NEXT: seq
// DIALECT-NEXT: staticlogic
// DIALECT-NEXT: std
// DIALECT-NEXT: sv
