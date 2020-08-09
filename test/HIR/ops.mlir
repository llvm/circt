// RUN: circt-opt %s | FileCheck %s
module {
  func @foo() {
    %t = hir.def_time_var : !hir.time
    %t2 = hir.duplicate_time_var %t: !hir.time -> !hir.time
    hir.sync_time(%t, %t2):(!hir.time,!hir.time)
    return
  }
}
