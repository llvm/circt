// RUN: circt-opt %s --convert-comb-to-arith | FileCheck %s

// CHECK-LABEL: @test
hw.module @test(in %arg0: i32, in %arg1: i32, in %arg2: i32, in %arg3: i32, in %arg4: i1) {
  // CHECK-NEXT: %c42_i32 = arith.constant 42 : i32
  %c42_i32 = hw.constant 42 : i32

  // CHECK-DAG: [[IS_ZERO:%.+]] = arith.cmpi eq, %arg1, %c0_i32{{.*}}
  // CHECK-DAG: [[IS_INT_MIN:%.+]] = arith.cmpi eq, %arg0, %c-2147483648_i32{{.*}}
  // CHECK-DAG: [[IS_NEG_ONE:%.+]] = arith.cmpi eq, %arg1, %c-1_i32{{.*}}
  // CHECK-DAG: [[IS_OVERFLOW:%.+]] = arith.andi [[IS_INT_MIN]], [[IS_NEG_ONE]]{{.*}}
  // CHECK-DAG: [[PRED:%.+]] = arith.ori [[IS_ZERO]], [[IS_OVERFLOW]]{{.*}}
  // CHECK-DAG: [[SAFE_DIVISOR:%.+]] = arith.select [[PRED]], %c1_i32{{.*}}, %arg1
  // CHECK-DAG: [[DIV_RESULT:%.+]] = arith.divsi %arg0, [[SAFE_DIVISOR]] : i32
  // CHECK-DAG: arith.select [[IS_OVERFLOW]], %c-2147483648_i32{{.*}}, [[DIV_RESULT]] : i32
  %0 = comb.divs %arg0, %arg1 : i32
  // CHECK: [[IS_ZERO:%.+]] = arith.cmpi eq, %arg1, %c0_i32{{.*}}
  // CHECK-NEXT: [[DIVISOR:%.+]] = arith.select [[IS_ZERO]], %c1_i32{{.*}}, %arg1
  // CHECK-NEXT: arith.divui %arg0, [[DIVISOR]] : i32
  %1 = comb.divu %arg0, %arg1 : i32
  // CHECK-DAG: [[IS_ZERO:%.+]] = arith.cmpi eq, %arg1, %c0_i32{{.*}}
  // CHECK-DAG: [[IS_INT_MIN:%.+]] = arith.cmpi eq, %arg0, %c-2147483648_i32{{.*}}
  // CHECK-DAG: [[IS_NEG_ONE:%.+]] = arith.cmpi eq, %arg1, %c-1_i32{{.*}}
  // CHECK-DAG: [[IS_OVERFLOW:%.+]] = arith.andi [[IS_INT_MIN]], [[IS_NEG_ONE]]{{.*}}
  // CHECK-DAG: [[PRED:%.+]] = arith.ori [[IS_ZERO]], [[IS_OVERFLOW]]{{.*}}
  // CHECK-DAG: [[SAFE_DIVISOR:%.+]] = arith.select [[PRED]], %c1_i32{{.*}}, %arg1
  // CHECK-DAG: [[DIV_RESULT:%.+]] = arith.remsi %arg0, [[SAFE_DIVISOR]] : i32
  // CHECK-DAG: arith.select [[IS_OVERFLOW]], %c0_i32{{.*}}, [[DIV_RESULT]] : i32
  %2 = comb.mods %arg0, %arg1 : i32
  // CHECK: [[IS_ZERO:%.+]] = arith.cmpi eq, %arg1, %c0_i32{{.*}}
  // CHECK-NEXT: [[DIVISOR:%.+]] = arith.select [[IS_ZERO]], %c1_i32{{.*}}, %arg1
  // CHECK-NEXT: arith.remui %arg0, [[DIVISOR]] : i32
  %3 = comb.modu %arg0, %arg1 : i32

  // CHECK-NEXT: arith.subi %arg0, %arg1 : i32
  %7 = comb.sub %arg0, %arg1 : i32

  // CHECK-NEXT: [[A1:%.+]] = arith.addi %arg0, %arg1 : i32
  // CHECK-NEXT: [[A2:%.+]] = arith.addi [[A1]], %arg2 : i32
  // CHECK-NEXT: arith.addi [[A2]], %arg3 : i32
  %8 = comb.add %arg0, %arg1, %arg2, %arg3 : i32
  // CHECK-NEXT: [[B1:%.+]] = arith.muli %arg0, %arg1 : i32
  // CHECK-NEXT: [[B2:%.+]] = arith.muli [[B1]], %arg2 : i32
  // CHECK-NEXT: arith.muli [[B2]], %arg3 : i32
  %9 = comb.mul %arg0, %arg1, %arg2, %arg3 : i32
  // CHECK-NEXT: [[C1:%.+]] = arith.andi %arg0, %arg1 : i32
  // CHECK-NEXT: [[C2:%.+]] = arith.andi [[C1]], %arg2 : i32
  // CHECK-NEXT: arith.andi [[C2]], %arg3 : i32
  %10 = comb.and %arg0, %arg1, %arg2, %arg3 : i32
  // CHECK-NEXT: [[D1:%.+]] = arith.ori %arg0, %arg1 : i32
  // CHECK-NEXT: [[D2:%.+]] = arith.ori [[D1]], %arg2 : i32
  // CHECK-NEXT: arith.ori [[D2]], %arg3 : i32
  %11 = comb.or %arg0, %arg1, %arg2, %arg3 : i32
  // CHECK-NEXT: [[E1:%.+]] = arith.xori %arg0, %arg1 : i32
  // CHECK-NEXT: [[E2:%.+]] = arith.xori [[E1]], %arg2 : i32
  // CHECK-NEXT: arith.xori [[E2]], %arg3 : i32
  %12 = comb.xor %arg0, %arg1, %arg2, %arg3 : i32

  // CHECK-NEXT: arith.select %arg4, %arg0, %arg1 : i32
  %13 = comb.mux %arg4, %arg0, %arg1 : i32

  // CHECK-NEXT: arith.cmpi eq, %arg0, %arg1 : i32
  %14 = comb.icmp eq %arg0, %arg1 : i32
  // CHECK-NEXT: arith.cmpi ne, %arg0, %arg1 : i32
  %15 = comb.icmp ne %arg0, %arg1 : i32
  // CHECK-NEXT: arith.cmpi eq, %arg0, %arg1 : i32
  %16 = comb.icmp ceq %arg0, %arg1 : i32
  // CHECK-NEXT: arith.cmpi ne, %arg0, %arg1 : i32
  %17 = comb.icmp cne %arg0, %arg1 : i32
  // CHECK-NEXT: arith.cmpi eq, %arg0, %arg1 : i32
  %18 = comb.icmp weq %arg0, %arg1 : i32
  // CHECK-NEXT: arith.cmpi ne, %arg0, %arg1 : i32
  %19 = comb.icmp wne %arg0, %arg1 : i32
  // CHECK-NEXT: arith.cmpi sle, %arg0, %arg1 : i32
  %20 = comb.icmp sle %arg0, %arg1 : i32
  // CHECK-NEXT: arith.cmpi slt, %arg0, %arg1 : i32
  %21 = comb.icmp slt %arg0, %arg1 : i32
  // CHECK-NEXT: arith.cmpi ule, %arg0, %arg1 : i32
  %22 = comb.icmp ule %arg0, %arg1 : i32
  // CHECK-NEXT: arith.cmpi ult, %arg0, %arg1 : i32
  %23 = comb.icmp ult %arg0, %arg1 : i32
  // CHECK-NEXT: arith.cmpi sge, %arg0, %arg1 : i32
  %24 = comb.icmp sge %arg0, %arg1 : i32
  // CHECK-NEXT: arith.cmpi sgt, %arg0, %arg1 : i32
  %25 = comb.icmp sgt %arg0, %arg1 : i32
  // CHECK-NEXT: arith.cmpi uge, %arg0, %arg1 : i32
  %26 = comb.icmp uge %arg0, %arg1 : i32
  // CHECK-NEXT: arith.cmpi ugt, %arg0, %arg1 : i32
  %27 = comb.icmp ugt %arg0, %arg1 : i32

  // CHECK-NEXT: %c5_i32 = arith.constant 5 : i32
  // CHECK-NEXT: [[V0:%.+]] = arith.shrui %arg0, %c5_i32 : i32
  // CHECK-NEXT: arith.trunci [[V0]] : i32 to i16
  %28 = comb.extract %arg0 from 5 : (i32) -> i16

  // CHECK-NEXT: [[AGG0:%.+]] = arith.extui %arg1 : i32 to i64
  // CHECK-NEXT: [[C32:%.+]] = arith.constant 32 : i64
  // CHECK-NEXT: [[V0:%.+]] = arith.extui %arg0 : i32 to i64
  // CHECK-NEXT: [[V1:%.+]] = arith.shli [[V0]], [[C32]] : i64
  // CHECK-NEXT: [[AGG1:%.+]] = arith.ori [[AGG0]], [[V1]] : i64
  %29 = comb.concat %arg0, %arg1 : i32, i32

  // CHECK-NEXT: arith.extsi %arg4 : i1 to i32
  %30 = comb.replicate %arg4 : (i1) -> i32

  // CHECK-NEXT: [[AGG0:%.+]] = arith.extui %arg0 : i32 to i64
  // CHECK-NEXT: [[C32:%.+]] = arith.constant 32 : i64
  // CHECK-NEXT: [[V0:%.+]] = arith.extui %arg0 : i32 to i64
  // CHECK-NEXT: [[V1:%.+]] = arith.shli [[V0]], [[C32]] : i64
  // CHECK-NEXT: [[AGG1:%.+]] = arith.ori [[AGG0]], [[V1]] : i64
  %31 = comb.replicate %arg0 : (i32) -> i64

  // The following used to trigger the arith.extui verifier.
  // CHECK-NEXT: scf.execute_region
  // CHECK-NEXT: scf.yield %arg0 : i32
  scf.execute_region -> i32 {
    %32 = comb.concat %arg0 : i32
    scf.yield %32 : i32
  }
}

// CHECK-LABEL: @shlTest
hw.module @shlTest(in %arg0: i32, in %arg1: i32) {
  // CHECK-NEXT: [[CST0:%.+]] = arith.constant 0 : i32
  // CHECK-NEXT: [[CST32:%.+]] = arith.constant 32 : i32
  // CHECK-NEXT: [[V0:%.+]] = arith.shli %arg0, %arg1 : i32
  // CHECK-NEXT: [[V1:%.+]] = arith.cmpi uge, %arg1, [[CST32]] : i32
  // CHECK-NEXT: [[V2:%.+]] = arith.select [[V1]], [[CST0]], [[V0]] : i32
  %shl = comb.shl %arg0, %arg1 : i32
}

// CHECK-LABEL: @shruTest
hw.module @shruTest(in %arg0: i32, in %arg1: i32) {
  // CHECK-NEXT: [[CST0:%.+]] = arith.constant 0 : i32
  // CHECK-NEXT: [[CST32:%.+]] = arith.constant 32 : i32
  // CHECK-NEXT: [[V0:%.+]] = arith.shrui %arg0, %arg1 : i32
  // CHECK-NEXT: [[V1:%.+]] = arith.cmpi uge, %arg1, [[CST32]] : i32
  // CHECK-NEXT: [[V2:%.+]] = arith.select [[V1]], [[CST0]], [[V0]] : i32
  %shru = comb.shru %arg0, %arg1 : i32
}

// CHECK-LABEL: @shrsTest
hw.module @shrsTest(in %arg0: i32, in %arg1: i32) {
  // CHECK-NEXT: [[CST31:%.+]] = arith.constant 31 : i32
  // CHECK-NEXT: [[V0:%.+]] = arith.minui %arg1, [[CST31]] : i32
  // CHECK-NEXT: [[V1:%.+]] = arith.shrsi %arg0, [[V0]] : i32
  %shrs = comb.shrs %arg0, %arg1 : i32
}
