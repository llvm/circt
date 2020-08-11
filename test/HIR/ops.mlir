// RUN: circt-opt %s | FileCheck %s
func @foo() {
  %t = hir.def_time_var : !hir.time
  %t2 = hir.duplicate_time_var %t : !hir.time
  hir.sync_time(%t, %t2) : (!hir.time,!hir.time)

  %l = "dummy_op"() : () -> (i32)
  %u = "dummy_op"() : () -> (i32)
  %s = "dummy_op"() : () -> (i32)
  %ts = "dummy_op"() : () -> (i32)
  hir.for %i = %l to %u step %s iter_time(%ti = %t tstep %ts):i32{
    // CHECK: %[[ADDR:.*]] = "dummy_op"() 
    %x = "dummy_op"() : () -> (i32)
    // CHECK: %[[MEM:.*]] = "mem_def"() 
    %A = "mem_def"() : ()->(!hir.mem_interface)
    // CHECK: {{%.*}} = hir.mem_read %[[MEM]]{{\[}}%[[ADDR]]{{\]}} at %{{.*}}
    %v = hir.mem_read %A[%x] at  %ti : !hir.mem_interface -> i32
  }
  %y = "dummy_op"() : () -> (i32)
  hir.for %j = %l to %u step %s iter_time(%tj = %t):i32{
    %A = "mem_def"() : ()->(!hir.mem_interface)
    %v = hir.mem_read %A[%y] at  %tj : !hir.mem_interface -> i32
  }

  //FIXME: If I use %x instead of %s or %ts etc., we get core dump instead of
  //use not dominating def error. This issue only occurs if %x doesnt have any
  //other use like below.
  hir.for %i = %l to %u step %s iter_time(%ti = %t tstep %ts):i32{
    %x = "dummy_op"() : () -> (i32)
  }
  return
}
