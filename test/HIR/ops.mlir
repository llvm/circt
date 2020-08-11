// RUN: circt-opt %s | FileCheck %s
func @foo() {
  %t = hir.def_time_var : !hir.time
  %t2 = hir.duplicate_time_var %t : !hir.time -> !hir.time
  hir.sync_time(%t, %t2) : (!hir.time,!hir.time)
  // CHECK: %[[ADDR:.*]] = "dummy_op"() 
  %x = "dummy_op"() : () -> (i32)
  // CHECK: %[[MEM:.*]] = "mem_def"() 
  %A = "mem_def"() : ()->(!hir.mem_interface)
  // CHECK: {{%.*}} = hir.mem_read %[[MEM]]{{\[}}%[[ADDR]]{{\]}} at %{{.*}}
  %v = hir.mem_read %A[%x] at  %t : !hir.mem_interface -> i32
  return
}
