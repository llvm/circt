// RUN: circt-opt --moore-pull-nonblocking-up %s | FileCheck %s

// CHECK-LABEL: moore.module @Foo(in %clk : !moore.l1)
moore.module @Foo(in %clk : !moore.l1) {
  %clk_0 = moore.net name "clk" wire : <l1>
  %0 = moore.constant 0 : i8
  %1 = moore.conversion %0 : !moore.i8 -> !moore.l8
  %2 = moore.conversion %1 : !moore.l8 -> !moore.l4
  %arr = moore.variable %2 : <l4>
  moore.procedure always {
    %3 = moore.read %clk_0 : l1
    moore.wait_event posedge %3 : l1
    // CHECK: %4 = moore.constant 1 : i32
    // CHECK: %5 = moore.bool_cast %4 : i32 -> i1
    %4 = moore.constant 1 : i32
    %5 = moore.bool_cast %4 : i32 -> i1
    %6 = moore.conversion %5 : !moore.i1 -> i1
    
    // CHECK-NOT: scf.if
    scf.if %6 {
    // CHECK-NOT: else
    } else {
      // CHECK: %7 = moore.constant 2 : i32
      // CHECK: %8 = moore.bool_cast %7 : i32 -> i1
      %7 = moore.constant 2 : i32
      %8 = moore.bool_cast %7 : i32 -> i1
      %9 = moore.conversion %8 : !moore.i1 -> i1
      // CHECK-NOT: scf.if
      scf.if %9 {
        %13 = moore.constant 0 : i32
        // CHECK: %11 = moore.extract_ref %arr
        %14 = moore.extract_ref %arr from %13 : <l4>, i32 -> <l1>
        %15 = moore.constant true : i1
        %16 = moore.conversion %15 : !moore.i1 -> !moore.l1

        // CHECK: %14 = moore.not %5 : i1
        // CHECK: %15 = moore.and %8, %14 : i1
        // CHECK: moore.nonblocking_assign %11, %13 if %15 : l1, !moore.i1
        moore.nonblocking_assign %14, %16 : l1
        %17 = moore.constant 1 : i32
        // CHECK: %17 = moore.extract_ref %arr
        %18 = moore.extract_ref %arr from %17 : <l4>, i32 -> <l1>
        %19 = moore.constant true : i1
        %20 = moore.conversion %19 : !moore.i1 -> !moore.l1

        // CHECK: %20 = moore.not %5 : i1
        // CHECK: %21 = moore.and %8, %20 : i1
        // CHECK: moore.nonblocking_assign %17, %19 if %21 : l1, !moore.i1
        moore.nonblocking_assign %18, %20 : l1
      }
      // CHECK: %22 = moore.constant 3 : i32
      // CHECK: %23 = moore.bool_cast %22 : i32 -> i1
      %10 = moore.constant 3 : i32
      %11 = moore.bool_cast %10 : i32 -> i1
      %12 = moore.conversion %11 : !moore.i1 -> i1
      // CHECK-NOT: scf.if
      scf.if %12 {
        %13 = moore.constant 2 : i32
        // CHECK: %26 = moore.extract_ref %arr
        %14 = moore.extract_ref %arr from %13 : <l4>, i32 -> <l1>
        %15 = moore.constant true : i1
        %16 = moore.conversion %15 : !moore.i1 -> !moore.l1

        // CHECK: %29 = moore.not %5 : i1
        // CHECK: %30 = moore.and %23, %29 : i1
        // CHECK: moore.nonblocking_assign %26, %28 if %30 : l1, !moore.i1
        moore.nonblocking_assign %14, %16 : l1
        %17 = moore.constant 3 : i32
        // CHECK: %32 = moore.extract_ref %arr
        %18 = moore.extract_ref %arr from %17 : <l4>, i32 -> <l1>
        %19 = moore.constant true : i1
        %20 = moore.conversion %19 : !moore.i1 -> !moore.l1

        // CHECK: %35 = moore.not %5 : i1
        // CHECK: %36 = moore.and %23, %35 : i1
        // CHECK: moore.nonblocking_assign %32, %34 if %36 : l1, !moore.i1 
        moore.nonblocking_assign %18, %20 : l1
      }
    }
  }
  // CHECK: moore.assign %clk_0, %clk : l1
  // CHECK: moore.output

  moore.assign %clk_0, %clk : l1
  moore.output
}
