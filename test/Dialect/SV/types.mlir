// RUN: circt-opt %s | circt-opt | FileCheck %s

// CHECK: sv.typedef @foo : i1
sv.typedef @foo : i1
