// RUN: circt-opt %s | FileCheck %s
func @foo() {
  %t = hir.def_time_var : !hir.time
  %t2 = hir.duplicate_time_var %t : !hir.time
  hir.sync_time(%t, %t2) : (!hir.time,!hir.time)

  %l = "dummy_op"() : () -> (!hir.int)
  %u = "dummy_op"() : () -> (!hir.int)
  %s = "dummy_op"() : () -> (!hir.int)
  %ts = "dummy_op"() : () -> (!hir.int)
  hir.for %i = %l to %u step %s iter_time(%ti = %t tstep %ts) : !hir.int{
    // CHECK: %[[ADDR:.*]] = "dummy_op"() 
    %x = "dummy_op"() : () -> (!hir.int)
    // CHECK: %[[MEM:.*]] = "mem_def"() 
    %A = "mem_def"() : ()->(!hir.mem_interface)
    // CHECK: {{%.*}} = hir.mem_read %[[MEM]]{{\[}}%[[ADDR]]{{\]}} at %{{.*}}
    %v = hir.mem_read %A[%x] at  %ti delay 10: !hir.mem_interface -> !hir.int
  }
  %y = "dummy_op"() : () -> (!hir.int)
  hir.for %j = %l to %u step %s iter_time(%tj = %t):!hir.int{
    %A = "mem_def"() : ()->(!hir.mem_interface)
    %v = hir.mem_read %A[%y] at  %tj : !hir.mem_interface -> !hir.int
  }

  //FIXME: If I use %x instead of %s or %ts etc., we get core dump instead of
  //use not dominating def error. This issue only occurs if %x doesnt have any
  //other use like below.
  hir.for %i = %l to %u step %s iter_time(%ti = %t tstep %ts):!hir.int{
    %x = "dummy_op"() : () -> (!hir.int)
  }
  %z1 = hir.call @foo (%l,%u) at %t:(!hir.int,!hir.int)->(!hir.int)
  %z2 = hir.call @bar (%l,%u) at %t delay 3:(!hir.int,!hir.int)->(!hir.int)
  
  return
}
