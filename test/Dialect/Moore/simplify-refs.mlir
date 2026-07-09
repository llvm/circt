// RUN: circt-opt --moore-simplify-refs %s | FileCheck %s

// CHECK-LABEL: moore.module @Foo()
 moore.module @Foo() {
    %a = moore.variable : <i8960>
    %b = moore.variable : <i42>
    %c = moore.variable : <i9002>
    %u = moore.variable : <l8960>
    %v = moore.variable : <l42>
    %w = moore.variable : <l9002>

    // CHECK-NOT: moore.concat_ref %a, %b
    %0 = moore.concat_ref %a, %b : (!moore.ref<i8960>, !moore.ref<i42>) -> <i9002>
    // CHECK: %[[C_READ:.+]] = moore.read %c : <i9002>
    %1 = moore.read %c : <i9002>
    // CHECK: %[[TMP1:.+]] = moore.extract %[[C_READ]] from 42 : i9002 -> i8960
    // CHECK: moore.assign %a, %[[TMP1]] : i8960
    // CHECK: %[[TMP2:.+]] = moore.extract %[[C_READ]] from 0 : i9002 -> i42
    // CHECK: moore.assign %b, %[[TMP2]] : i42
    moore.assign %0, %1 : i9002
    moore.procedure always {
      // CHECK-NOT: moore.concat_ref %u, %v
      %2 = moore.concat_ref %u, %v : (!moore.ref<l8960>, !moore.ref<l42>) -> <l9002>
      // CHECK: %[[W_READ:.+]] = moore.read %w : <l9002>
      // CHECK: %[[TMP1:.+]] = moore.extract %[[W_READ]] from 42 : l9002 -> l8960
      // CHECK: moore.blocking_assign %u, %[[TMP1]] : l8960
      // CHECK: %[[TMP2:.+]] = moore.extract %[[W_READ]] from 0 : l9002 -> l42
      // CHECK: moore.blocking_assign %v, %[[TMP2]] : l42
      %3 = moore.read %w : <l9002>
      moore.blocking_assign %2, %3 : l9002

      %4 = moore.constant 1 : i32
      %5 = moore.bool_cast %4 : i32 -> i1
      %6 = moore.to_builtin_int %5 : !moore.i1
      scf.if %6 {
        // CHECK-NOT: moore.concat_ref %u, %v
        %7 = moore.concat_ref %u, %v : (!moore.ref<l8960>, !moore.ref<l42>) -> <l9002>
        // CHECK: %[[W_READ:.+]] = moore.read %w : <l9002>
        %8 = moore.read %w : <l9002>
        // CHECK: %[[TMP1:.+]] = moore.extract %[[W_READ]] from 42 : l9002 -> l8960
        // CHECK: moore.nonblocking_assign %u, %[[TMP1]] : l8960
        // CHECK: %[[TMP2:.+]] = moore.extract %[[W_READ]] from 0 : l9002 -> l42
        // CHECK: moore.nonblocking_assign %v, %[[TMP2]] : l42
        moore.nonblocking_assign %7, %8 : l9002
      }
      moore.return
    }
    moore.output
  }

// CHECK-LABEL: moore.module @Nested()
moore.module @Nested() {
  %x = moore.variable : <i32>
  %y = moore.variable : <i32>
  %z = moore.variable : <i32>
  moore.procedure always {
    // CHECK: %[[Z_READ:.+]] = moore.read %z
    %4 = moore.read %z : <i32>
    // CHECK: %[[TMP2:.+]] = moore.zext %[[Z_READ]] : i32 -> i96
    %6 = moore.zext %4 : !moore.i32 -> !moore.i96
    
    // CHECK-NOT: moore.concat_ref %x, %x
    %0 = moore.concat_ref %x, %x : (!moore.ref<i32>, !moore.ref<i32>) -> <i64>
    %1 = moore.concat_ref %0 : (!moore.ref<i64>) -> <i64>
    %2 = moore.concat_ref %y : (!moore.ref<i32>) -> <i32>
    %3 = moore.concat_ref %1, %2 : (!moore.ref<i64>, !moore.ref<i32>) -> <i96>
    
    // CHECK: %[[TMP3:.+]] = moore.extract %[[TMP2]] from 64 : i96 -> i32
    // CHECK: moore.blocking_assign %x, %[[TMP3]] : i32
    // CHECK: %[[TMP4:.+]] = moore.extract %[[TMP2]] from 32 : i96 -> i32
    // CHECK: moore.blocking_assign %x, %[[TMP4]] : i32
    // CHECK: %[[TMP5:.+]] = moore.extract %[[TMP2]] from 0 : i96 -> i32
    // CHECK: moore.blocking_assign %y, %[[TMP5]] : i32
    moore.blocking_assign %3, %6 : i96
    moore.return
  }
  moore.output
}

// Delayed assignment variants keep their delay on every slice
// (`{a, b} <= #1 c;` and `assign #1 {a, b} = c;`).
// CHECK-LABEL: moore.module @DelayedConcatDst()
moore.module @DelayedConcatDst() {
  %a = moore.variable : <l7>
  %b = moore.variable : <l1>
  %c = moore.variable : <l8>
  %d = moore.variable : <l8>
  // CHECK: [[T0:%.+]] = moore.constant_time 2000000 fs
  %t0 = moore.constant_time 2000000 fs
  // CHECK-NOT: moore.concat_ref
  %2 = moore.concat_ref %a, %b : (!moore.ref<l7>, !moore.ref<l1>) -> <l8>
  // CHECK: %[[D_READ:.+]] = moore.read %d : <l8>
  %3 = moore.read %d : <l8>
  // CHECK: %[[SL1:.+]] = moore.extract %[[D_READ]] from 1 : l8 -> l7
  // CHECK: moore.delayed_assign %a, %[[SL1]], [[T0]] : l7
  // CHECK: %[[SL2:.+]] = moore.extract %[[D_READ]] from 0 : l8 -> l1
  // CHECK: moore.delayed_assign %b, %[[SL2]], [[T0]] : l1
  moore.delayed_assign %2, %3, %t0 : l8
  moore.procedure always {
    // CHECK: [[T1:%.+]] = moore.constant_time 1000000 fs
    %t = moore.constant_time 1000000 fs
    // CHECK-NOT: moore.concat_ref
    %0 = moore.concat_ref %a, %b : (!moore.ref<l7>, !moore.ref<l1>) -> <l8>
    // CHECK: %[[C_READ:.+]] = moore.read %c : <l8>
    %1 = moore.read %c : <l8>
    // CHECK: %[[TMP1:.+]] = moore.extract %[[C_READ]] from 1 : l8 -> l7
    // CHECK: moore.delayed_nonblocking_assign %a, %[[TMP1]], [[T1]] : l7
    // CHECK: %[[TMP2:.+]] = moore.extract %[[C_READ]] from 0 : l8 -> l1
    // CHECK: moore.delayed_nonblocking_assign %b, %[[TMP2]], [[T1]] : l1
    moore.delayed_nonblocking_assign %0, %1, %t : l8
    moore.return
  }
  moore.output
}

// CHECK-LABEL: moore.module @QueueRefs()
moore.module @QueueRefs() {
  %q = moore.variable : <queue<i32, 5>>
  moore.procedure initial {
    %0 = moore.constant 0 : i32
    %1 = moore.constant 1 : i32

    // CHECK: moore.queue.set %q[%0] = %1 : <queue<i32, 5>>
    %el = moore.dyn_queue_ref_element %q from %0 : <queue<i32, 5>>, i32 -> <i32>
    moore.blocking_assign %el, %1 : i32
    moore.return
  }
}

// CHECK-LABEL: moore.module @QueueRefsWithConcat()
moore.module @QueueRefsWithConcat() {
  %q = moore.variable : <queue<i32, 5>>
  moore.procedure initial {
    // CHECK: [[ONE:%.+]] = moore.constant 1 : i32
    %one = moore.constant 1 : i32
    // CHECK: [[TWO:%.+]] = moore.constant 2 : i32
    %two = moore.constant 2 : i32

    %el1 = moore.dyn_queue_ref_element %q from %one : <queue<i32, 5>>, i32 -> <i32>
    %el2 = moore.dyn_queue_ref_element %q from %two : <queue<i32, 5>>, i32 -> <i32>

    // CHECK: [[CCAT:%.+]] = moore.concat [[ONE]], [[TWO]] : (!moore.i32, !moore.i32) -> i64
    %ccat = moore.concat %one, %two : (!moore.i32, !moore.i32) -> i64
    %bothEls = moore.concat_ref %el1, %el2 : (!moore.ref<i32>, !moore.ref<i32>) -> <i64>

    // CHECK: [[EXTR1:%.+]] = moore.extract [[CCAT]] from 32 : i64 -> i32
    // CHECK: moore.queue.set %q[[[ONE]]] = [[EXTR1]] : <queue<i32, 5>>
    // CHECK: [[EXTR2:%.+]] = moore.extract [[CCAT]] from 0 : i64 -> i32
    // CHECK: moore.queue.set %q[[[TWO]]] = [[EXTR2]] : <queue<i32, 5>>
    moore.blocking_assign %bothEls, %ccat : i64
    moore.return
  }
}
