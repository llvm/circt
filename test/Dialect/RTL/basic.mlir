// RUN: circt-opt %s -verify-diagnostics | circt-opt -verify-diagnostics | FileCheck %s

// CHECK-LABEL: rtl.module @test1(%arg0: i3, %arg1: i1, %arg2: !rtl.array<1000xi8>) -> (i50) {
rtl.module @test1(%arg0: i3, %arg1: i1, %arg2: !rtl.array<1000xi8>) -> (i50) {
  // CHECK-NEXT:    %c42_i12 = comb.constant 42 : i12
  // CHECK-NEXT:    [[RES0:%[0-9]+]] = comb.add %c42_i12, %c42_i12 : i12
  // CHECK-NEXT:    [[RES1:%[0-9]+]] = comb.mul %c42_i12, [[RES0]] : i12
  %a = comb.constant 42 : i12
  %b = comb.add %a, %a : i12
  %c = comb.mul %a, %b : i12

  // CHECK-NEXT:    [[RES2:%[0-9]+]] = comb.sext %arg0 : (i3) -> i7
  %d = comb.sext %arg0 : (i3) -> i7

  // CHECK-NEXT:    [[RES4:%[0-9]+]] = comb.concat %c42_i12 : (i12) -> i12
  %conc1 = comb.concat %a : (i12) -> i12

  // CHECK-NEXT:    [[RES7:%[0-9]+]] = comb.xorr [[RES4]] : i12
  %xorr1 = comb.xorr %conc1 : i12

  // CHECK-NEXT:    [[RES8:%[0-9]+]] = comb.concat [[RES4]], [[RES0]], [[RES1]], [[RES2]], [[RES2]] : (i12, i12, i12, i7, i7) -> i50
  %result = comb.concat %conc1, %b, %c, %d, %d : (i12, i12, i12, i7, i7) -> i50

  // CHECK-NEXT: [[RES9:%[0-9]+]] = comb.extract [[RES8]] from 4 : (i50) -> i19
  %small1 = comb.extract %result from 4 : (i50) -> i19

  // CHECK-NEXT: [[RES10:%[0-9]+]] = comb.extract [[RES8]] from 31 : (i50) -> i19
  %small2 = comb.extract %result from 31 : (i50) -> i19

  // CHECK-NEXT: comb.add [[RES9]], [[RES10]] : i19
  %add = comb.add %small1, %small2 : i19

  // CHECK-NEXT: comb.icmp eq [[RES9]], [[RES10]] : i19
  %eq = comb.icmp eq %small1, %small2 : i19

  // CHECK-NEXT: comb.icmp ne [[RES9]], [[RES10]] : i19
  %neq = comb.icmp ne %small1, %small2 : i19

  // CHECK-NEXT: comb.icmp slt [[RES9]], [[RES10]] : i19
  %lt = comb.icmp slt %small1, %small2 : i19

  // CHECK-NEXT: comb.icmp ult [[RES9]], [[RES10]] : i19
  %ult = comb.icmp ult %small1, %small2 : i19

  // CHECK-NEXT: comb.icmp sle [[RES9]], [[RES10]] : i19
  %leq = comb.icmp sle %small1, %small2 : i19

  // CHECK-NEXT: comb.icmp ule [[RES9]], [[RES10]] : i19
  %uleq = comb.icmp ule %small1, %small2 : i19

  // CHECK-NEXT: comb.icmp sgt [[RES9]], [[RES10]] : i19
  %gt = comb.icmp sgt %small1, %small2 : i19

  // CHECK-NEXT: comb.icmp ugt [[RES9]], [[RES10]] : i19
  %ugt = comb.icmp ugt %small1, %small2 : i19

  // CHECK-NEXT: comb.icmp sge [[RES9]], [[RES10]] : i19
  %geq = comb.icmp sge %small1, %small2 : i19

  // CHECK-NEXT: comb.icmp uge [[RES9]], [[RES10]] : i19
  %ugeq = comb.icmp uge %small1, %small2 : i19

  // CHECK-NEXT: %w = sv.wire : !rtl.inout<i4>
  %w = sv.wire : !rtl.inout<i4>

  // CHECK-NEXT: %after1 = sv.wire : !rtl.inout<i4>
  %before1 = sv.wire {name = "after1"} : !rtl.inout<i4>

  // CHECK-NEXT: sv.read_inout %after1 : !rtl.inout<i4>
  %read_before1 = sv.read_inout %before1 : !rtl.inout<i4>

  // CHECK-NEXT: %after2_conflict = sv.wire : !rtl.inout<i4>
  // CHECK-NEXT: %after2_conflict_0 = sv.wire {name = "after2_conflict"} : !rtl.inout<i4>
  %before2_0 = sv.wire {name = "after2_conflict"} : !rtl.inout<i4>
  %before2_1 = sv.wire {name = "after2_conflict"} : !rtl.inout<i4>

  // CHECK-NEXT: %after3 = sv.wire {someAttr = "foo"} : !rtl.inout<i4>
  %before3 = sv.wire {name = "after3", someAttr = "foo"} : !rtl.inout<i4>

  // CHECK-NEXT: = comb.mux %arg1, [[RES2]], [[RES2]] : i7
  %mux = comb.mux %arg1, %d, %d : i7
  
  // CHECK-NEXT: [[STR:%[0-9]+]] = rtl.struct_create ({{.*}}, {{.*}}) : !rtl.struct<foo: i19, bar: i7>
  %s0 = rtl.struct_create (%small1, %mux) : !rtl.struct<foo: i19, bar: i7>

  // CHECK-NEXT: = rtl.struct_extract [[STR]]["foo"] : !rtl.struct<foo: i19, bar: i7>
  %sf1 = rtl.struct_extract %s0["foo"] : !rtl.struct<foo: i19, bar: i7>

  // CHECK-NEXT: = rtl.struct_inject [[STR]]["foo"], {{.*}} : !rtl.struct<foo: i19, bar: i7>
  %s1 = rtl.struct_inject %s0["foo"], %sf1 : !rtl.struct<foo: i19, bar: i7>
  
  // CHECK-NEXT: :2 = rtl.struct_explode [[STR]] : !rtl.struct<foo: i19, bar: i7>
  %se:2 = rtl.struct_explode %s0 : !rtl.struct<foo: i19, bar: i7>

  // CHECK-NEXT: comb.bitcast [[STR]] : (!rtl.struct<foo: i19, bar: i7>)
  %structBits = comb.bitcast %s0 : (!rtl.struct<foo: i19, bar: i7>) -> i26

  // CHECK-NEXT: = constant 13 : i10
  %idx = constant 13 : i10
  // CHECK-NEXT: = rtl.array_slice %arg2 at %c13_i10 : (!rtl.array<1000xi8>) -> !rtl.array<24xi8>
  %subArray = rtl.array_slice %arg2 at %idx : (!rtl.array<1000xi8>) -> !rtl.array<24xi8>
  // CHECK-NEXT: [[ARR1:%.+]] = rtl.array_create [[RES9]], [[RES10]] : (i19)
  %arrCreated = rtl.array_create %small1, %small2 : (i19)
  // CHECK-NEXT: [[ARR2:%.+]] = rtl.array_create [[RES9]], [[RES10]], {{.+}} : (i19)
  %arr2 = rtl.array_create %small1, %small2, %add : (i19)
  // CHECK-NEXT: = rtl.array_concat [[ARR1]], [[ARR2]] : !rtl.array<2xi19>, !rtl.array<3xi19>
  %bigArray = rtl.array_concat %arrCreated, %arr2 : !rtl.array<2 x i19>, !rtl.array<3 x i19>

  // CHECK-NEXT:    rtl.output [[RES8]] : i50
  rtl.output %result : i50
}
// CHECK-NEXT:  }
