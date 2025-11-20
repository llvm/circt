// RUN: circt-opt %s --verify-roundtrip | FileCheck %s
// RUN: circt-opt %s  --lower-esi-ports --lower-esi-to-hw --lower-esi-types --verify-roundtrip | FileCheck %s --check-prefix=LOW


!TypeA = !hw.struct<header1: i6, header2: i1, header3: !hw.array<13xi16>>
!TypeAwin1 = !esi.window<
  "TypeAwin1", !TypeA, [
    <"FrameA", [
      <"header1">,
      <"header2">
    ]>,
    <"FrameB", [
      <"header3", 3>
    ]>
  ]>


// CHECK-LABEL:   hw.module.extern @TypeAModuleDst(in %windowed : !esi.window<"TypeAwin1", !hw.struct<header1: i6, header2: i1, header3: !hw.array<13xi16>>, [<"FrameA", [<"header1">, <"header2">]>, <"FrameB", [<"header3", 3>]>]>)
hw.module.extern @TypeAModuleDst(in %windowed: !TypeAwin1)
// CHECK-LABEL:   hw.module.extern @TypeAModuleSrc(out windowed : !esi.window<"TypeAwin1", !hw.struct<header1: i6, header2: i1, header3: !hw.array<13xi16>>, [<"FrameA", [<"header1">, <"header2">]>, <"FrameB", [<"header3", 3>]>]>)
hw.module.extern @TypeAModuleSrc(out windowed: !TypeAwin1)

!lowered = !hw.union<
  FrameA: !hw.struct<header1: i6, header2: i1>,
  FrameB: !hw.struct<header3: !hw.array<3xi16>>,
  FrameB_leftOver: !hw.struct<header3: !hw.array<1xi16>>>

// CHECK-LABEL: hw.module @TypeAModuleUnwrap
// CHECK:         [[r0:%.+]] = esi.window.unwrap %a : !esi.window<"TypeAwin1", !hw.struct<header1: i6, header2: i1, header3: !hw.array<13xi16>>, [<"FrameA", [<"header1">, <"header2">]>, <"FrameB", [<"header3", 3>]>]>
// CHECK:         hw.output [[r0]] : !hw.union<FrameA: !hw.struct<header1: i6, header2: i1>, FrameB: !hw.struct<header3: !hw.array<3xi16>>, FrameB_leftOver: !hw.struct<header3: !hw.array<1xi16>>>
hw.module @TypeAModuleUnwrap(in %a: !TypeAwin1, out x: !lowered) {
  %u = esi.window.unwrap %a : !TypeAwin1
  hw.output %u : !lowered
}

// CHECK-LABEL: hw.module @TypeAModuleUnwrapWrap
// CHECK:         [[r0:%.+]] = esi.window.unwrap %a : !esi.window<"TypeAwin1", !hw.struct<header1: i6, header2: i1, header3: !hw.array<13xi16>>, [<"FrameA", [<"header1">, <"header2">]>, <"FrameB", [<"header3", 3>]>]>
// CHECK:         [[r1:%.+]] = esi.window.wrap [[r0]] : !esi.window<"TypeAwin1", !hw.struct<header1: i6, header2: i1, header3: !hw.array<13xi16>>, [<"FrameA", [<"header1">, <"header2">]>, <"FrameB", [<"header3", 3>]>]>
// LOW-LABEL: hw.module @TypeAModuleUnwrapWrap
// LOW-NEXT:    hw.output %a
hw.module @TypeAModuleUnwrapWrap(in %a: !TypeAwin1, out x: !TypeAwin1) {
  %u = esi.window.unwrap %a : !TypeAwin1
  %x = esi.window.wrap %u : !TypeAwin1
  hw.output %x : !TypeAwin1
}

// LOW-LABEL: hw.module @TypeAModuleWrapUnwrap
// LOW-NEXT:    hw.output %a
hw.module @TypeAModuleWrapUnwrap(in %a: !lowered, out x: !lowered) {
  %w = esi.window.wrap %a : !TypeAwin1
  %u = esi.window.unwrap %w : !TypeAwin1
  hw.output %u : !lowered
}

// LOW-LABEL:  hw.module @TypeAModulePassthrough(in %a : !hw.union<FrameA: !hw.struct<header1: i6, header2: i1>, FrameB: !hw.struct<header3: !hw.array<3xi16>>, FrameB_leftOver: !hw.struct<header3: !hw.array<1xi16>>>, out x : !hw.union<FrameA: !hw.struct<header1: i6, header2: i1>, FrameB: !hw.struct<header3: !hw.array<3xi16>>, FrameB_leftOver: !hw.struct<header3: !hw.array<1xi16>>>) {
// LOW-NEXT:     %foo.x = hw.instance "foo" @TypeAModuleUnwrapWrap(a: %a: !hw.union<FrameA: !hw.struct<header1: i6, header2: i1>, FrameB: !hw.struct<header3: !hw.array<3xi16>>, FrameB_leftOver: !hw.struct<header3: !hw.array<1xi16>>>) -> (x: !hw.union<FrameA: !hw.struct<header1: i6, header2: i1>, FrameB: !hw.struct<header3: !hw.array<3xi16>>, FrameB_leftOver: !hw.struct<header3: !hw.array<1xi16>>>)
// LOW-NEXT:     hw.output %foo.x :  !hw.union<FrameA: !hw.struct<header1: i6, header2: i1>, FrameB: !hw.struct<header3: !hw.array<3xi16>>, FrameB_leftOver: !hw.struct<header3: !hw.array<1xi16>>>
hw.module @TypeAModulePassthrough(in %a: !TypeAwin1, out x: !TypeAwin1) {
  %x = hw.instance "foo" @TypeAModuleUnwrapWrap(a: %a: !TypeAwin1) -> (x: !TypeAwin1)
  hw.output %x : !TypeAwin1
}

!ListType = !hw.struct<data: !esi.list<i32>>
!ListTypeWin_one = !esi.window<"ListTypeWin", !ListType, [
  <"", [
    <"data">
  ]>
]>
!ListTypeWin_four = !esi.window<"ListTypeWin", !ListType, [
  <"DataFrame", [
    <"data", 4>
  ]>
]>
// LOW-LABEL:   hw.module @ListModule_one(in %windowed_in : !hw.struct<data: i32, last: i1>, out windowed_out : !hw.struct<data: i32, last: i1>) {
// LOW-NEXT:     %data, %last = hw.struct_explode %windowed_in : !hw.struct<data: i32, last: i1>
// LOW-NEXT:     %c1_i32 = hw.constant 1 : i32
// LOW-NEXT:     [[R0:%.+]] = comb.add %data, %c1_i32 : i32
// LOW-NEXT:     [[R1:%.+]] = hw.struct_create ([[R0]], %last) : !hw.struct<data: i32, last: i1>
// LOW-NEXT:     hw.output [[R1]] : !hw.struct<data: i32, last: i1>
hw.module @ListModule_one(in %windowed_in: !ListTypeWin_one, out windowed_out: !ListTypeWin_one) {
  %data_struct = esi.window.unwrap %windowed_in : !ListTypeWin_one
  %data, %last = hw.struct_explode %data_struct : !hw.struct<data: i32, last: i1>
  %c1 = hw.constant 1 : i32
  %data_inc = comb.add %data, %c1 : i32
  %out_struct = hw.struct_create (%data_inc, %last) : !hw.struct<data: i32, last: i1>
  %out = esi.window.wrap %out_struct : !ListTypeWin_one
  hw.output %out: !ListTypeWin_one
}

// LOW-LABEL:  hw.module @ListModule_four(in %windowed_in : !hw.union<DataFrame: !hw.struct<data: !hw.array<4xi32>, data_size: i2, last: i1>>, out windowed_out : !hw.union<DataFrame: !hw.struct<data: !hw.array<4xi32>, data_size: i2, last: i1>>) {
// LOW-NEXT:     hw.output %windowed_in : !hw.union<DataFrame: !hw.struct<data: !hw.array<4xi32>, data_size: i2, last: i1>>
hw.module @ListModule_four(in %windowed_in: !ListTypeWin_four, out windowed_out: !ListTypeWin_four) {
  %data_struct = esi.window.unwrap %windowed_in : !ListTypeWin_four
  %out = esi.window.wrap %data_struct : !ListTypeWin_four
  hw.output %out: !ListTypeWin_four
}

// LOW-LABEL:  hw.module @ListChannelModule(in %in : !hw.struct<data: i32, last: i1>, in %in_valid : i1, in %out_ready : i1, out in_ready : i1, out out : !hw.struct<data: i32, last: i1>, out out_valid : i1) {
// LOW-NEXT:     %data, %last = hw.struct_explode %in : !hw.struct<data: i32, last: i1>
// LOW-NEXT:     %c1_i32 = hw.constant 1 : i32
// LOW-NEXT:     [[R0:%.+]] = comb.add %data, %c1_i32 : i32
// LOW-NEXT:     [[R1:%.+]] = hw.struct_create ([[R0]], %last) : !hw.struct<data: i32, last: i1>
// LOW-NEXT:     hw.output %out_ready, [[R1]], %in_valid : i1, !hw.struct<data: i32, last: i1>, i1
hw.module @ListChannelModule(in %in: !esi.channel<!ListTypeWin_one>, out out: !esi.channel<!ListTypeWin_one>) {
  %windowed_in, %in_valid = esi.unwrap.vr %in, %ready : !ListTypeWin_one
  %data_struct = esi.window.unwrap %windowed_in : !ListTypeWin_one
  %data, %last = hw.struct_explode %data_struct : !hw.struct<data: i32, last: i1>
  %c1 = hw.constant 1 : i32
  %data_inc = comb.add %data, %c1 : i32
  %out_struct = hw.struct_create (%data_inc, %last) : !hw.struct<data: i32, last: i1>
  %out = esi.window.wrap %out_struct : !ListTypeWin_one
  %channel_out, %ready = esi.wrap.vr %out, %in_valid : !ListTypeWin_one
  hw.output %channel_out : !esi.channel<!ListTypeWin_one>
}

!arrStruct = !hw.struct<field1: !hw.array<5xi32>, field2: i8>
!arrStructWindow = !esi.window<"window1", !arrStruct, [
  <"frame1", [<"field1", 5>, <"field2">]>
]>

hw.module @ArrayStructWindowModule(in %a: !arrStructWindow, out x: !hw.union<frame1: !hw.struct<field1: !hw.array<5xi32>, field2: i8>>) {
  %u = esi.window.unwrap %a : !arrStructWindow
  hw.output %u : !hw.union<
    frame1: !hw.struct<field1: !hw.array<5xi32>, field2: i8>>
}

// LOW-LABEL: hw.module @ArrayStructWindowModule
// LOW-NEXT:    hw.output %a


!TypeBugFix = !hw.struct<header1: i6, header2: i1, header3: !hw.array<13xi16>>
!TypeBugFixWin = !esi.window<
  "TypeBugFixWin", !TypeBugFix, [
    <"FrameA", [
      <"header1">,
      <"header3", 3>,
      <"header2">
    ]>
  ]>

// LOW-LABEL: hw.module.extern @TypeBugFixModuleDst
// LOW-SAME: !hw.union<FrameA: !hw.struct<header1: i6, header3: !hw.array<3xi16>, header2: i1>, FrameA_leftOver: !hw.struct<header1: i6, header3: !hw.array<1xi16>, header2: i1>>
hw.module.extern @TypeBugFixModuleDst(in %windowed: !TypeBugFixWin)
