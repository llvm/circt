// RUN: circt-opt %s | FileCheck %s
module {
  func @foo() {
    %t = hir.def_time_var : !hir.time
    %t2 = hir.duplicate_time_var %t: !hir.time -> !hir.time
    hir.sync_time(%t, %t2):(!hir.time,!hir.time)
    // CHECK: [[ADDR:%[0-9]]] = "dummy_op"() 
    %x = "dummy_op"() : () -> (i32)
    // CHECK: [[MEM:%[0-9]]] = "mem_def"() 
    %A = "mem_def"():()->(!hir.mem_interface)
    // CHECK: {{%[0-9]}} = hir.mem_read [[MEM]]{{\[}}[[ADDR]]{{\]}} at %0 
    %v = hir.mem_read %A[%x] at  %t: !hir.mem_interface -> i32
    return
  }
}
