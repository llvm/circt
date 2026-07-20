// RUN: circt-opt %s --lower-seq-to-sv | FileCheck %s

hw.module @CommentAttrs(in %clk : !seq.clock, in %in : i1, out out : i1) {
  // CHECK: %reg = sv.reg {comment = "register comment"} : !hw.inout<i1>
  %reg = seq.firreg %in clock %clk {comment = "register comment"} : i1
  hw.output %reg : i1
}
