// RUN: circt-opt --symbol-dce %s | FileCheck %s

// CHECK-NOT: @seq1
rtg.sequence @seq1() {
  %str = rtg.constant "to be removed" : !rtg.string
  rtg.comment %str
}
