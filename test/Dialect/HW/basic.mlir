// RUN: circt-opt %s --verify-diagnostics --verify-roundtrip | FileCheck %s

// CHECK-LABEL: hw.module @test1(in %arg0 : i3, in %arg1 : i1, in %arg2 : !hw.array<1000xi8>, out result : i50) {
hw.module @test1(in %arg0: i3, in %arg1: i1, in %arg2: !hw.array<1000xi8>, out result: i50) {
  // CHECK-NEXT:    %c42_i12 = hw.constant 42 : i12
  // CHECK-NEXT:    [[RES0:%[0-9]+]] = comb.add %c42_i12, %c42_i12 : i12
  // CHECK-NEXT:    [[RES1:%[0-9]+]] = comb.mul %c42_i12, [[RES0]] : i12
  %a = hw.constant 42 : i12
  %b = comb.add %a, %a : i12
  %c = comb.mul %a, %b : i12

  // CHECK-NEXT:    [[RES2:%[0-9]+]] = comb.concat %arg0, %arg0, %arg1
  %d = comb.concat %arg0, %arg0, %arg1 : i3, i3, i1

  // CHECK-NEXT:    [[RES4:%[0-9]+]] = comb.concat %c42_i12 : i12
  %conc1 = comb.concat %a : i12

  // CHECK-NEXT:    [[RES7:%[0-9]+]] = comb.parity [[RES4]] : i12
  %parity1 = comb.parity %conc1 : i12

  // CHECK-NEXT:    [[RES8:%[0-9]+]] = comb.concat [[RES4]], [[RES0]], [[RES1]], [[RES2]], [[RES2]] : i12, i12, i12, i7, i7
  %result = comb.concat %conc1, %b, %c, %d, %d : i12, i12, i12, i7, i7

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

  // CHECK-NEXT: %w = sv.wire : !hw.inout<i4>
  %w = sv.wire : !hw.inout<i4>

  // CHECK-NEXT: %after1 = sv.wire : !hw.inout<i4>
  %before1 = sv.wire name "after1" : !hw.inout<i4>

  // CHECK-NEXT: sv.read_inout %after1 : !hw.inout<i4>
  %read_before1 = sv.read_inout %before1 : !hw.inout<i4>

  // CHECK-NEXT: %after2_conflict = sv.wire : !hw.inout<i4>
  // CHECK-NEXT: %after2_conflict_0 = sv.wire name "after2_conflict" : !hw.inout<i4>
  %before2_0 = sv.wire name "after2_conflict" : !hw.inout<i4>
  %before2_1 = sv.wire name "after2_conflict" : !hw.inout<i4>

  // CHECK-NEXT: %after3 = sv.wire {someAttr = "foo"} : !hw.inout<i4>
  %before3 = sv.wire name "after3" {someAttr = "foo"} : !hw.inout<i4>

  // CHECK-NEXT: %w2 = hw.wire [[RES2]] : i7
  %w2 = hw.wire %d : i7

  // CHECK-NEXT: %after4 = hw.wire [[RES2]] : i7
  %before4 = hw.wire %d name "after4" : i7

  // CHECK-NEXT: %after5_conflict = hw.wire [[RES2]] : i7
  // CHECK-NEXT: %after5_conflict_1 = hw.wire [[RES2]] name "after5_conflict" : i7
  %before5_0 = hw.wire %d name "after5_conflict" : i7
  %before5_1 = hw.wire %d name "after5_conflict" : i7

  // CHECK-NEXT: %after6 = hw.wire [[RES2]] {someAttr = "foo"} : i7
  %before6 = hw.wire %d name "after6" {someAttr = "foo"} : i7

  // CHECK-NEXT: = comb.mux %arg1, [[RES2]], [[RES2]] : i7
  %mux = comb.mux %arg1, %d, %d : i7

  // CHECK-NEXT: [[STR:%[0-9]+]] = hw.struct_create ({{.*}}, {{.*}}) : !hw.struct<foo: i19, bar: i7>
  %s0 = hw.struct_create (%small1, %mux) : !hw.struct<foo: i19, bar: i7>

  // CHECK-NEXT: %foo = hw.struct_extract [[STR]]["foo"] : !hw.struct<foo: i19, bar: i7>
  %foo = hw.struct_extract %s0["foo"] : !hw.struct<foo: i19, bar: i7>

  // CHECK-NEXT: = hw.struct_inject [[STR]]["foo"], {{.*}} : !hw.struct<foo: i19, bar: i7>
  %s1 = hw.struct_inject %s0["foo"], %foo : !hw.struct<foo: i19, bar: i7>

  // CHECK-NEXT:  %foo_2, %bar = hw.struct_explode [[STR]] : !hw.struct<foo: i19, bar: i7>
  %foo_2, %bar = hw.struct_explode %s0 : !hw.struct<foo: i19, bar: i7>

  // CHECK-NEXT: hw.bitcast [[STR]] : (!hw.struct<foo: i19, bar: i7>)
  %structBits = hw.bitcast %s0 : (!hw.struct<foo: i19, bar: i7>) -> i26

  // CHECK-NEXT: = arith.constant 13 : i10
  %idx = arith.constant 13 : i10
  // CHECK-NEXT: = hw.array_slice %arg2[%c13_i10] : (!hw.array<1000xi8>) -> !hw.array<24xi8>
  %subArray = hw.array_slice %arg2[%idx] : (!hw.array<1000xi8>) -> !hw.array<24xi8>
  // CHECK-NEXT: [[ARR1:%.+]] = hw.array_create [[RES9]], [[RES10]] : i19
  %arrCreated = hw.array_create %small1, %small2 : i19
  // CHECK-NEXT: [[ARR2:%.+]] = hw.array_create [[RES9]], [[RES10]], {{.+}} : i19
  %arr2 = hw.array_create %small1, %small2, %add : i19
  // CHECK-NEXT: = hw.array_concat [[ARR1]], [[ARR2]] : !hw.array<2xi19>, !hw.array<3xi19>
  %bigArray = hw.array_concat %arrCreated, %arr2 : !hw.array<2 x i19>, !hw.array<3 x i19>
  // CHECK-NEXT: %A = hw.enum.constant A : !hw.enum<A, B, C>
  %A_enum = hw.enum.constant A : !hw.enum<A, B, C>
  // CHECK-NEXT: %B = hw.enum.constant B : !hw.enum<A, B, C>
  %B_enum = hw.enum.constant B : !hw.enum<A, B, C>
  // CHECK-NEXT: = hw.enum.cmp %A, %B : !hw.enum<A, B, C>, !hw.enum<A, B, C>
  %enumcmp = hw.enum.cmp %A_enum, %B_enum : !hw.enum<A, B, C>, !hw.enum<A, B, C>

  // CHECK-NEXT: hw.aggregate_constant [false, true] : !hw.struct<a: i1, b: i1>
  hw.aggregate_constant [false, true] : !hw.struct<a: i1, b: i1>
  //hw.enum.constant A : !hw.enum<A, B>
  // CHECK-NEXT: hw.aggregate_constant [0 : i2, 1 : i2, -2 : i2, -1 : i2] : !hw.array<4xi2>
  hw.aggregate_constant [0 : i2, 1 : i2, -2 : i2, -1 : i2] : !hw.array<4xi2>
  // CHECK-NEXT: hw.aggregate_constant [false] : !hw.uarray<1xi1>
  hw.aggregate_constant [false] : !hw.uarray<1xi1>
  // CHECK-NEXT{LITERAL}: hw.aggregate_constant [[false]] : !hw.struct<a: !hw.array<1xi1>>
  hw.aggregate_constant [[false]] : !hw.struct<a: !hw.array<1xi1>>
  // CHECK-NEXT: hw.aggregate_constant ["A"] : !hw.struct<a: !hw.enum<A, B, C>>
  hw.aggregate_constant ["A"] : !hw.struct<a: !hw.enum<A, B, C>>
  // CHECK-NEXT: hw.aggregate_constant ["A"] : !hw.array<1xenum<A, B, C>>
  hw.aggregate_constant ["A"] : !hw.array<1 x!hw.enum<A, B, C>>

  // CHECK-NEXT:    hw.output [[RES8]] : i50
  hw.output %result : i50
}
// CHECK-NEXT:  }

func.func @ArrayOps(%a: !hw.array<1000xi42>, %i: i10, %v: i42) {
  hw.array_inject %a[%i], %v : !hw.array<1000xi42>, i10
  return
}

hw.module @UnionOps(in %a: !hw.union<foo: i1, bar: i3>, out x: i3, out z: !hw.union<bar: i3, baz: i8>) {
  %x = hw.union_extract %a["bar"] : !hw.union<foo: i1, bar: i3>
  %z = hw.union_create "bar", %x : !hw.union<bar: i3, baz: i8>
  hw.output %x, %z : i3, !hw.union<bar: i3, baz: i8>
}
// CHECK-LABEL: hw.module @UnionOps(in %a : !hw.union<foo: i1, bar: i3>, out x : i3, out z : !hw.union<bar: i3, baz: i8>) {
// CHECK-NEXT:    [[I3REG:%.+]] = hw.union_extract %a["bar"] : !hw.union<foo: i1, bar: i3>
// CHECK-NEXT:    [[UREG:%.+]] = hw.union_create "bar", [[I3REG]] : !hw.union<bar: i3, baz: i8>
// CHECK-NEXT:    hw.output [[I3REG]], [[UREG]] : i3, !hw.union<bar: i3, baz: i8>

// https://github.com/llvm/circt/issues/863
// CHECK-LABEL: hw.module @signed_arrays
hw.module @signed_arrays(in %arg0: si8, out out: !hw.array<2xsi8>) {
  // CHECK-NEXT:  %wireArray = sv.wire  : !hw.inout<array<2xsi8>>
  %wireArray = sv.wire : !hw.inout<!hw.array<2xsi8>>

  // CHECK-NEXT: %0 = hw.array_create %arg0, %arg0 : si8
  %0 = hw.array_create %arg0, %arg0 : si8

  // CHECK-NEXT: sv.assign %wireArray, %0 : !hw.array<2xsi8>
  sv.assign %wireArray, %0 : !hw.array<2xsi8>

  %result = sv.read_inout %wireArray : !hw.inout<!hw.array<2xsi8>>
  hw.output %result : !hw.array<2xsi8>
}

// Check that we pass the verifier that the module's function type matches
// the block argument types when using InOutTypes.
// CHECK: hw.module @InOutPort(inout %arg0 : i1)
hw.module @InOutPort(inout %arg0: i1) { }

/// Port names that aren't valid MLIR identifiers are handled with `argNames`
/// attribute being explicitly printed.
// https://github.com/llvm/circt/issues/1822

// CHECK-LABEL: hw.module @argRenames
// CHECK-SAME: attributes {argNames = [""]}
hw.module @argRenames(in %arg1: i32) attributes {argNames = [""]} {
}

// CHECK-LABEL: hw.module @commentModule
// CHECK-SAME: attributes {comment = "hello world"}
hw.module @commentModule() attributes {comment = "hello world"} {}

 // CHECK-LABEL: hw.module @Foo(in %_d2A " d*" : i1, in %0 "" : i1, out "" : i1, out "1" : i1) {
hw.module @Foo(in %0 " d*" : i1, in %1 "" : i1, out "" : i1, out "1" : i1) {
  hw.output %0 , %1: i1, i1
 }
 // CHECK-LABEL: hw.module @Bar(in %foo : i1, out "" : i1, out "" : i1) {
hw.module @Bar(in %foo : i1, out "" : i1, out "" : i1) {
   // CHECK-NEXT:  hw.instance "foo" @Foo(" d*": %foo: i1, "": %foo: i1) -> ("": i1, "1": i1)
  %0, %1 = hw.instance "foo" @Foo(" d*": %foo: i1, "": %foo : i1)  -> ("" : i1, "1" : i1)
  hw.output %0, %1 : i1, i1
}

// CHECK-LABEL: hw.module @bitwise(in %arg0 : i7, in %arg1 : i7, out r : i21) {
hw.module @bitwise(in %arg0: i7, in %arg1: i7, out r: i21) {
// CHECK-NEXT:    [[RES0:%[0-9]+]] = comb.xor %arg0 : i7
// CHECK-NEXT:    [[RES1:%[0-9]+]] = comb.or  %arg0, %arg1 : i7
// CHECK-NEXT:    [[RES2:%[0-9]+]] = comb.and %arg0, %arg1, %arg0 : i7
  %and1 = comb.xor %arg0 : i7
  %or1  = comb.or  %arg0, %arg1 : i7
  %xor1 = comb.and %arg0, %arg1, %arg0 : i7

// CHECK-NEXT:    [[RESULT:%[0-9]+]] = comb.concat [[RES0]], [[RES1]], [[RES2]] : i7, i7, i7
// CHECK-NEXT:    hw.output [[RESULT]] : i21
  %result = comb.concat %and1, %or1, %xor1 : i7, i7, i7
  hw.output %result : i21
}

// CHECK-LABEL: hw.module @shl_op(in %a : i7, in %b : i7, out r : i7) {
hw.module @shl_op(in %a: i7, in %b: i7, out r: i7) {
// CHECK-NEXT:    [[RES:%[0-9]+]] = comb.shl  %a, %b : i7
  %0  = comb.shl  %a, %b : i7
// CHECK-NEXT:    hw.output [[RES]]
  hw.output %0 : i7
}

// CHECK-LABEL: hw.module @shr_op(in %a : i7, in %b : i7, out r0 : i7) {
hw.module @shr_op(in %a: i7, in %b: i7, out r0: i7) {
// CHECK-NEXT:    [[RES:%[0-9]+]] = comb.shru %a, %b : i7
  %0  = comb.shru %a, %b : i7
// CHECK-NEXT:    hw.output [[RES]]
  hw.output %0 : i7
}

// CHECK-LABEL: hw.module @casts(in %in1 : i7, out x : !hw.struct<int: i7>)
hw.module @casts(in %in1: i7, out x: !hw.struct<int: i7>) {
  // CHECK-NEXT: %0 = hw.bitcast %in1 : (i7) -> !hw.array<7xi1>
  // CHECK-NEXT: %1 = hw.bitcast %0 : (!hw.array<7xi1>) -> !hw.struct<int: i7>
  %bits = hw.bitcast %in1 : (i7) -> !hw.array<7xi1>
  %backToInt = hw.bitcast %bits : (!hw.array<7xi1>) -> !hw.struct<int: i7>
  hw.output %backToInt : !hw.struct<int: i7>
}

hw.module private @TargetA(in %a: i32, out b: i32) {
  hw.output %a : i32
}

hw.module private @TargetB(in %a: i32, out b: i32) {
  hw.output %a : i32
}

hw.module private @TargetDefault(in %a: i32, out b: i32) {
  hw.output %a : i32
}

hw.module public @top(in %a: i32) {
  // CHECK: hw.instance_choice "inst1" sym @inst1 option "bar" @TargetDefault or @TargetA if "A" or @TargetB if "B"(a: %a: i32) -> (b: i32)
  hw.instance_choice "inst1" sym @inst1 option "bar" @TargetDefault or @TargetA if "A" or @TargetB if "B"(a: %a: i32) -> (b: i32)
  // CHECK: hw.instance_choice "inst2" option "baz" @TargetDefault(a: %a: i32) -> (b: i32)
  hw.instance_choice "inst2" option "baz" @TargetDefault(a: %a: i32) -> (b: i32)
}

// CHECK-LABEL: @aggregate_const
hw.module @aggregate_const(out o : !hw.array<1x!seq.clock>) {
  // CHECK-NEXT: hw.aggregate_constant [#seq<clock_constant high> : !seq.clock] : !hw.array<1x!seq.clock>
  %0 = hw.aggregate_constant [#seq<clock_constant high> : !seq.clock] : !hw.array<1x!seq.clock>
  hw.output %0 : !hw.array<1x!seq.clock>
}
 hw.module @hiimped(in %a : !hw.hiz<i3>, in %b : i3, out %c : i3) {
  %d = hw.hiz.read %a : i3
 }