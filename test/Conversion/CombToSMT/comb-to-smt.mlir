// RUN: circt-opt %s --convert-comb-to-smt | FileCheck %s

// CHECK-LABEL: func @test
// CHECK-SAME: ([[A0:%.+]]: !smt.bv<32>, [[A1:%.+]]: !smt.bv<32>, [[A2:%.+]]: !smt.bv<32>, [[A3:%.+]]: !smt.bv<32>, [[A4:%.+]]: !smt.bv<1>, [[ARG5:%.+]]: !smt.bv<4>)
func.func @test(%a0: !smt.bv<32>, %a1: !smt.bv<32>, %a2: !smt.bv<32>, %a3: !smt.bv<32>, %a4: !smt.bv<1>, %a5: !smt.bv<4>) {
  %arg0 = builtin.unrealized_conversion_cast %a0 : !smt.bv<32> to i32
  %arg1 = builtin.unrealized_conversion_cast %a1 : !smt.bv<32> to i32
  %arg2 = builtin.unrealized_conversion_cast %a2 : !smt.bv<32> to i32
  %arg3 = builtin.unrealized_conversion_cast %a3 : !smt.bv<32> to i32
  %arg4 = builtin.unrealized_conversion_cast %a4 : !smt.bv<1> to i1
  %arg5 = builtin.unrealized_conversion_cast %a5 : !smt.bv<4> to i4

  // CHECK:      [[ZERO:%.+]] = smt.bv.constant #smt.bv<0> : !smt.bv<32>
  // CHECK-NEXT: [[IS_ZERO:%.+]] = smt.eq [[A1]], [[ZERO]] : !smt.bv<32>
  // CHECK-NEXT: [[UNDEF:%.+]] = smt.declare_fun : !smt.bv<32>
  // CHECK-NEXT: [[DIV:%.+]] = smt.bv.sdiv [[A0]], [[A1]] : !smt.bv<32>
  // CHECK-NEXT: smt.ite [[IS_ZERO]], [[UNDEF]], [[DIV]] : !smt.bv<32>
  %0 = comb.divs %arg0, %arg1 : i32
  // CHECK-NEXT: [[ZERO:%.+]] = smt.bv.constant #smt.bv<0> : !smt.bv<32>
  // CHECK-NEXT: [[IS_ZERO:%.+]] = smt.eq [[A1]], [[ZERO]] : !smt.bv<32>
  // CHECK-NEXT: [[UNDEF:%.+]] = smt.declare_fun : !smt.bv<32>
  // CHECK-NEXT: [[DIV:%.+]] = smt.bv.udiv [[A0]], [[A1]] : !smt.bv<32>
  // CHECK-NEXT: smt.ite [[IS_ZERO]], [[UNDEF]], [[DIV]] : !smt.bv<32>
  %1 = comb.divu %arg0, %arg1 : i32
  // CHECK-NEXT: [[ZERO:%.+]] = smt.bv.constant #smt.bv<0> : !smt.bv<32>
  // CHECK-NEXT: [[IS_ZERO:%.+]] = smt.eq [[A1]], [[ZERO]] : !smt.bv<32>
  // CHECK-NEXT: [[UNDEF:%.+]] = smt.declare_fun : !smt.bv<32>
  // CHECK-NEXT: [[DIV:%.+]] = smt.bv.srem [[A0]], [[A1]] : !smt.bv<32>
  // CHECK-NEXT: smt.ite [[IS_ZERO]], [[UNDEF]], [[DIV]] : !smt.bv<32>
  %2 = comb.mods %arg0, %arg1 : i32
  // CHECK-NEXT: [[ZERO:%.+]] = smt.bv.constant #smt.bv<0> : !smt.bv<32>
  // CHECK-NEXT: [[IS_ZERO:%.+]] = smt.eq [[A1]], [[ZERO]] : !smt.bv<32>
  // CHECK-NEXT: [[UNDEF:%.+]] = smt.declare_fun : !smt.bv<32>
  // CHECK-NEXT: [[DIV:%.+]] = smt.bv.urem [[A0]], [[A1]] : !smt.bv<32>
  // CHECK-NEXT: smt.ite [[IS_ZERO]], [[UNDEF]], [[DIV]] : !smt.bv<32>
  %3 = comb.modu %arg0, %arg1 : i32

  // CHECK-NEXT: [[NEG:%.+]] = smt.bv.neg [[A1]] : !smt.bv<32>
  // CHECK-NEXT: smt.bv.add [[A0]], [[NEG]] : !smt.bv<32>
  %7 = comb.sub %arg0, %arg1 : i32

  // CHECK-NEXT: [[A5:%.+]] = smt.bv.add [[A0]], [[A1]] : !smt.bv<32>
  // CHECK-NEXT: [[A6:%.+]] = smt.bv.add [[A5]], [[A2]] : !smt.bv<32>
  // CHECK-NEXT: smt.bv.add [[A6]], [[A3]] : !smt.bv<32>
  %8 = comb.add %arg0, %arg1, %arg2, %arg3 : i32
  // CHECK-NEXT: [[B1:%.+]] = smt.bv.mul [[A0]], [[A1]] : !smt.bv<32>
  // CHECK-NEXT: [[B2:%.+]] = smt.bv.mul [[B1]], [[A2]] : !smt.bv<32>
  // CHECK-NEXT: smt.bv.mul [[B2]], [[A3]] : !smt.bv<32>
  %9 = comb.mul %arg0, %arg1, %arg2, %arg3 : i32
  // CHECK-NEXT: [[C1:%.+]] = smt.bv.and [[A0]], [[A1]] : !smt.bv<32>
  // CHECK-NEXT: [[C2:%.+]] = smt.bv.and [[C1]], [[A2]] : !smt.bv<32>
  // CHECK-NEXT: smt.bv.and [[C2]], [[A3]] : !smt.bv<32>
  %10 = comb.and %arg0, %arg1, %arg2, %arg3 : i32
  // CHECK-NEXT: [[D1:%.+]] = smt.bv.or [[A0]], [[A1]] : !smt.bv<32>
  // CHECK-NEXT: [[D2:%.+]] = smt.bv.or [[D1]], [[A2]] : !smt.bv<32>
  // CHECK-NEXT: smt.bv.or [[D2]], [[A3]] : !smt.bv<32>
  %11 = comb.or %arg0, %arg1, %arg2, %arg3 : i32
  // CHECK-NEXT: [[E1:%.+]] = smt.bv.xor [[A0]], [[A1]] : !smt.bv<32>
  // CHECK-NEXT: [[E2:%.+]] = smt.bv.xor [[E1]], [[A2]] : !smt.bv<32>
  // CHECK-NEXT: smt.bv.xor [[E2]], [[A3]] : !smt.bv<32>
  %12 = comb.xor %arg0, %arg1, %arg2, %arg3 : i32

  // CHECK-NEXT: [[CONST1:%.+]] = smt.bv.constant #smt.bv<-1> : !smt.bv<1>
  // CHECK-NEXT: [[COND:%.+]] = smt.eq [[A4]], [[CONST1]] : !smt.bv<1>
  // CHECK-NEXT: smt.ite [[COND]], [[A0]], [[A1]] : !smt.bv<32>
  %13 = comb.mux %arg4, %arg0, %arg1 : i32

  // CHECK-NEXT: smt.eq [[A0]], [[A1]] : !smt.bv<32>
  %14 = comb.icmp eq %arg0, %arg1 : i32
  // CHECK-NEXT: smt.distinct [[A0]], [[A1]] : !smt.bv<32>
  %15 = comb.icmp ne %arg0, %arg1 : i32
  // CHECK-NEXT: smt.bv.cmp sle [[A0]], [[A1]] : !smt.bv<32>
  %20 = comb.icmp sle %arg0, %arg1 : i32
  // CHECK-NEXT: smt.bv.cmp slt [[A0]], [[A1]] : !smt.bv<32>
  %21 = comb.icmp slt %arg0, %arg1 : i32
  // CHECK-NEXT: smt.bv.cmp ule [[A0]], [[A1]] : !smt.bv<32>
  %22 = comb.icmp ule %arg0, %arg1 : i32
  // CHECK-NEXT: smt.bv.cmp ult [[A0]], [[A1]] : !smt.bv<32>
  %23 = comb.icmp ult %arg0, %arg1 : i32
  // CHECK-NEXT: smt.bv.cmp sge [[A0]], [[A1]] : !smt.bv<32>
  %24 = comb.icmp sge %arg0, %arg1 : i32
  // CHECK-NEXT: smt.bv.cmp sgt [[A0]], [[A1]] : !smt.bv<32>
  %25 = comb.icmp sgt %arg0, %arg1 : i32
  // CHECK-NEXT: smt.bv.cmp uge [[A0]], [[A1]] : !smt.bv<32>
  %26 = comb.icmp uge %arg0, %arg1 : i32
  // CHECK-NEXT: smt.bv.cmp ugt [[A0]], [[A1]] : !smt.bv<32>
  %27 = comb.icmp ugt %arg0, %arg1 : i32

  // CHECK-NEXT: smt.bv.extract [[A0]] from 5 : (!smt.bv<32>) -> !smt.bv<16>
  %28 = comb.extract %arg0 from 5 : (i32) -> i16
  // CHECK-NEXT: smt.bv.concat [[A0]], [[A1]] : !smt.bv<32>, !smt.bv<32>
  %29 = comb.concat %arg0, %arg1 : i32, i32
  // CHECK-NEXT: smt.bv.repeat 32 times [[A4]] : !smt.bv<1>
  %30 = comb.replicate %arg4 : (i1) -> i32

  // CHECK-NEXT: %{{.*}} = smt.bv.shl [[A0]], [[A1]] : !smt.bv<32>
  %32 = comb.shl %arg0, %arg1 : i32
  // CHECK-NEXT: %{{.*}} = smt.bv.ashr [[A0]], [[A1]] : !smt.bv<32>
  %33 = comb.shrs %arg0, %arg1 : i32
  // CHECK-NEXT: %{{.*}} = smt.bv.lshr [[A0]], [[A1]] : !smt.bv<32>
  %34 = comb.shru %arg0, %arg1 : i32

  // The comb.icmp folder is called before the conversion patterns and produces
  // a `hw.constant` operation.
  // CHECK-NEXT: smt.bv.constant #smt.bv<-1> : !smt.bv<1>
  %35 = comb.icmp eq %arg0, %arg0 : i32

  // CHECK-NEXT: [[V0:%.+]] = smt.bv.extract [[ARG5]] from 0 : (!smt.bv<4>) -> !smt.bv<1>
  // CHECK-NEXT: [[V1:%.+]] = smt.bv.extract [[ARG5]] from 1 : (!smt.bv<4>) -> !smt.bv<1>
  // CHECK-NEXT: [[V2:%.+]] = smt.bv.xor [[V0]], [[V1]] : !smt.bv<1>
  // CHECK-NEXT: [[V3:%.+]] = smt.bv.extract [[ARG5]] from 2 : (!smt.bv<4>) -> !smt.bv<1>
  // CHECK-NEXT: [[V4:%.+]] = smt.bv.xor [[V2]], [[V3]] : !smt.bv<1>
  // CHECK-NEXT: [[V5:%.+]] = smt.bv.extract [[ARG5]] from 3 : (!smt.bv<4>) -> !smt.bv<1>
  // CHECK-NEXT: smt.bv.xor [[V4]], [[V5]] : !smt.bv<1>
  %36 = comb.parity %arg5 : i4

  return
}

// regression test for APInt comparison in BitVectorAttrStorage (just checking
// that it doesn't lead to a crash)
hw.module @decomposedShl(in %in1 : i2, in %in2 : i2, out out : i2) {
  %c0_i2 = hw.constant 0 : i2
  %0 = comb.icmp eq %c0_i2, %c0_i2 : i2
  hw.output %c0_i2 : i2
}
