// RUN: circt-opt --moore-simplify-procedures %s | FileCheck %s

// CHECK-LABEL: moore.module @Foo
moore.module @Foo() {
  %a = moore.variable : <i32>
  %x = moore.variable : <i32>
  %y = moore.variable : <i32>

  // CHECK: moore.procedure always_comb
  moore.procedure always_comb {
    // CHECK: [[TMP:%.+]] = moore.read %a
    // CHECK: [[LOCAL_A:%.+]] = moore.variable
    // CHECK: moore.blocking_assign [[LOCAL_A]], [[TMP]]

    // CHECK: [[C1:%.+]] = moore.constant 1
    // CHECK: moore.blocking_assign [[LOCAL_A]], [[C1]]
    // CHECK: [[TMP:%.+]] = moore.read [[LOCAL_A]]
    // CHECK: moore.blocking_assign %a, [[TMP]]
    %0 = moore.constant 1 : i32
    moore.blocking_assign %a, %0 : i32

    // CHECK: [[TMP:%.+]] = moore.read [[LOCAL_A]]
    // CHECK: moore.blocking_assign %x, [[TMP]]
    %1 = moore.read %a : <i32>
    moore.blocking_assign %x, %1 : i32

    // CHECK: [[TMP1:%.+]] = moore.read [[LOCAL_A]]
    // CHECK: [[TMP2:%.+]] = moore.constant 1
    // CHECK: [[TMP3:%.+]] = moore.add [[TMP1]], [[TMP2]]
    // CHECK: moore.blocking_assign [[LOCAL_A]], [[TMP3]]
    // CHECK: [[TMP:%.+]] = moore.read [[LOCAL_A]]
    // CHECK: moore.blocking_assign %a, [[TMP]]
    %2 = moore.read %a : <i32>
    %3 = moore.constant 1 : i32
    %4 = moore.add %2, %3 : i32
    moore.blocking_assign %a, %4 : i32

    moore.return
  }

  // CHECK: moore.procedure always_comb
  moore.procedure always_comb {
    // CHECK: [[TMP:%.+]] = moore.read %a
    // CHECK: [[LOCAL_A:%.+]] = moore.variable
    // CHECK: moore.blocking_assign [[LOCAL_A]], [[TMP]]

    // CHECK: [[TMP:%.+]] = moore.read [[LOCAL_A]]
    // CHECK: moore.blocking_assign %y, [[TMP]]
    %0 = moore.read %a : <i32>
    moore.blocking_assign %y, %0 : i32

    moore.return
  }
}
