// RUN: circt-opt --symbol-dce %s | FileCheck %s

// CHECK-NOT: @seq1
rtg.sequence @seq1() {
  rtg.comment "to be removed"
}
