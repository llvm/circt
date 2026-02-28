// RUN: circt-opt --moore-lower-concatref %s | FileCheck %s

// CHECK-LABEL: moore.module @Foo()
 moore.module @Foo() {
    %a = moore.variable : <i8960>
    %b = moore.variable : <i42>
    %c = moore.variable : <i9002>
    %u = moore.variable : <l8960>
    %v = moore.variable : <l42>
    %w = moore.variable : <l9002>

    // CHECK: moore.concat_ref %a, %b
    %0 = moore.concat_ref %a, %b : (!moore.ref<i8960>, !moore.ref<i42>) -> <i9002>
    // CHECK: %[[C_READ:.+]] = moore.read %c : <i9002>
    %1 = moore.read %c : <i9002>
    // CHECK: %[[TMP1:.+]] = moore.extract %[[C_READ]] from 42 : i9002 -> i8960
    // CHECK: moore.assign %a, %[[TMP1]] : i8960
    // CHECK: %[[TMP2:.+]] = moore.extract %[[C_READ]] from 0 : i9002 -> i42
    // CHECK: moore.assign %b, %[[TMP2]] : i42
    moore.assign %0, %1 : i9002
    moore.procedure always {
      // CHECK: moore.concat_ref %u, %v
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
        // CHECK: moore.concat_ref %u, %v
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
    
    // CHECK: moore.concat_ref %x, %x
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
