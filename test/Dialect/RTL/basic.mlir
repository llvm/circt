// RUN: circt-opt %s | FileCheck %s

// CHECK-LABEL: func @test1(%arg0: i3, %arg1: i1, %arg2: !rtl.array<1000xi8>) -> i50 {
func @test1(%arg0: i3, %arg1: i1, %arg2: !rtl.array<1000xi8>) -> i50 {
  // CHECK-NEXT:    %c42_i12 = rtl.constant(42 : i12) : i12
  // CHECK-NEXT:    [[RES0:%[0-9]+]] = rtl.add %c42_i12, %c42_i12 : i12
  // CHECK-NEXT:    [[RES1:%[0-9]+]] = rtl.mul %c42_i12, [[RES0]] : i12
  %a = rtl.constant(42 : i12) : i12
  %b = rtl.add %a, %a : i12
  %c = rtl.mul %a, %b : i12

  // CHECK-NEXT:    [[RES2:%[0-9]+]] = rtl.sext %arg0 : (i3) -> i7
  %d = rtl.sext %arg0 : (i3) -> i7

  // CHECK-NEXT:    [[RES4:%[0-9]+]] = rtl.concat %c42_i12 : (i12) -> i12
  %conc1 = rtl.concat %a : (i12) -> i12

  // CHECK-NEXT:    [[RES5:%[0-9]+]] = rtl.andr [[RES4]] : i12
  // CHECK-NEXT:    [[RES6:%[0-9]+]] = rtl.orr  [[RES4]] : i12
  // CHECK-NEXT:    [[RES7:%[0-9]+]] = rtl.xorr [[RES4]] : i12
  %andr1 = rtl.andr %conc1 : i12
  %orr1  = rtl.orr  %conc1 : i12
  %xorr1 = rtl.xorr %conc1 : i12

  // CHECK-NEXT:    [[RES8:%[0-9]+]] = rtl.concat [[RES4]], [[RES0]], [[RES1]], [[RES2]], [[RES2]] : (i12, i12, i12, i7, i7) -> i50
  %result = rtl.concat %conc1, %b, %c, %d, %d : (i12, i12, i12, i7, i7) -> i50

  // CHECK-NEXT: [[RES9:%[0-9]+]] = rtl.extract [[RES8]] from 4 : (i50) -> i19
  %small1 = rtl.extract %result from 4 : (i50) -> i19

  // CHECK-NEXT: [[RES10:%[0-9]+]] = rtl.extract [[RES8]] from 31 : (i50) -> i19
  %small2 = rtl.extract %result from 31 : (i50) -> i19

  // CHECK-NEXT: rtl.add [[RES9]], [[RES10]] : i19
  %add = rtl.add %small1, %small2 : i19

  // CHECK-NEXT: rtl.icmp eq [[RES9]], [[RES10]] : i19
  %eq = rtl.icmp eq %small1, %small2 : i19

  // CHECK-NEXT: rtl.icmp ne [[RES9]], [[RES10]] : i19
  %neq = rtl.icmp ne %small1, %small2 : i19

  // CHECK-NEXT: rtl.icmp slt [[RES9]], [[RES10]] : i19
  %lt = rtl.icmp slt %small1, %small2 : i19

  // CHECK-NEXT: rtl.icmp ult [[RES9]], [[RES10]] : i19
  %ult = rtl.icmp ult %small1, %small2 : i19

  // CHECK-NEXT: rtl.icmp sle [[RES9]], [[RES10]] : i19
  %leq = rtl.icmp sle %small1, %small2 : i19

  // CHECK-NEXT: rtl.icmp ule [[RES9]], [[RES10]] : i19
  %uleq = rtl.icmp ule %small1, %small2 : i19

  // CHECK-NEXT: rtl.icmp sgt [[RES9]], [[RES10]] : i19
  %gt = rtl.icmp sgt %small1, %small2 : i19

  // CHECK-NEXT: rtl.icmp ugt [[RES9]], [[RES10]] : i19
  %ugt = rtl.icmp ugt %small1, %small2 : i19

  // CHECK-NEXT: rtl.icmp sge [[RES9]], [[RES10]] : i19
  %geq = rtl.icmp sge %small1, %small2 : i19

  // CHECK-NEXT: rtl.icmp uge [[RES9]], [[RES10]] : i19
  %ugeq = rtl.icmp uge %small1, %small2 : i19

  // CHECK-NEXT: %w = rtl.wire : !rtl.inout<i4>
  %w = rtl.wire : !rtl.inout<i4>

  // CHECK-NEXT: %after1 = rtl.wire : !rtl.inout<i4>
  %before1 = rtl.wire {name = "after1"} : !rtl.inout<i4>

  // CHECK-NEXT: rtl.read_inout %after1 : !rtl.inout<i4>
  %read_before1 = rtl.read_inout %before1 : !rtl.inout<i4>

  // CHECK-NEXT: %after2_conflict = rtl.wire : !rtl.inout<i4>
  // CHECK-NEXT: %after2_conflict_0 = rtl.wire {name = "after2_conflict"} : !rtl.inout<i4>
  %before2_0 = rtl.wire {name = "after2_conflict"} : !rtl.inout<i4>
  %before2_1 = rtl.wire {name = "after2_conflict"} : !rtl.inout<i4>

  // CHECK-NEXT: %after3 = rtl.wire {someAttr = "foo"} : !rtl.inout<i4>
  %before3 = rtl.wire {name = "after3", someAttr = "foo"} : !rtl.inout<i4>

  // CHECK-NEXT: = rtl.mux %arg1, [[RES2]], [[RES2]] : i7
  %mux = rtl.mux %arg1, %d, %d : i7
  
  // CHECK-NEXT: [[STR:%[0-9]+]] = rtl.struct_create ({{.*}}, {{.*}}) : !rtl.struct<foo: i19, bar: i7>
  %s0 = rtl.struct_create (%small1, %mux) : !rtl.struct<foo: i19, bar: i7>
  
  // CHECK-NEXT: = rtl.struct_extract [[STR]]["foo"] : !rtl.struct<foo: i19, bar: i7>
  %sf1 = rtl.struct_extract %s0["foo"] : !rtl.struct<foo: i19, bar: i7>

  // CHECK-NEXT: = rtl.struct_inject [[STR]]["foo"], {{.*}} : !rtl.struct<foo: i19, bar: i7>
  %s1 = rtl.struct_inject %s0["foo"], %sf1 : !rtl.struct<foo: i19, bar: i7>
  
  // CHECK-NEXT: :2 = rtl.struct_explode [[STR]] : !rtl.struct<foo: i19, bar: i7>
  %se:2 = rtl.struct_explode %s0 : !rtl.struct<foo: i19, bar: i7>

  // CHECK-NEXT: rtl.bitcast [[STR]] : (!rtl.struct<foo: i19, bar: i7>)
  %structBits = rtl.bitcast %s0 : (!rtl.struct<foo: i19, bar: i7>) -> i26

  // CHECK-NEXT: = constant 13 : i10
  %idx = constant 13 : i10
  // CHECK-NEXT: = rtl.array_slice %arg2 at %c13_i10 : (!rtl.array<1000xi8>) -> !rtl.array<24xi8>
  %subArray = rtl.array_slice %arg2 at %idx : (!rtl.array<1000xi8>) -> !rtl.array<24xi8>

  // CHECK-NEXT:    return [[RES8]] : i50
  return %result : i50
}
// CHECK-NEXT:  }
