// RUN: circt-opt %s | FileCheck %s
hir.def @blah(%x,%y) at %t {time_domains = [1:i32,2:i32], 
  time_offsets = [2:i32,2:i32]}: (i32,i32)->(i32) {
}
