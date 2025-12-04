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


!ArrayWindow = !hw.array<2x!TypeAwin1>
!ListOfWindows = !esi.list<!TypeAwin1>
!AliasStruct = !hw.typealias<@AliasStruct, !hw.struct<aliasField: !TypeAwin1>>

// CHECK-LABEL:   hw.module.extern @TypeAModuleDst(in %windowed : !esi.window<"TypeAwin1", !hw.struct<header1: i6, header2: i1, header3: !hw.array<13xi16>>, [<"FrameA", [<"header1">, <"header2">]>, <"FrameB", [<"header3", 3>]>]>)
hw.module.extern @TypeAModuleDst(in %windowed: !TypeAwin1)
// CHECK-LABEL:   hw.module.extern @TypeAModuleSrc(out windowed : !esi.window<"TypeAwin1", !hw.struct<header1: i6, header2: i1, header3: !hw.array<13xi16>>, [<"FrameA", [<"header1">, <"header2">]>, <"FrameB", [<"header3", 3>]>]>)
hw.module.extern @TypeAModuleSrc(out windowed: !TypeAwin1)

// CHECK-LABEL: hw.module @ArrayWindowModule
// CHECK:         %[[C0:.*]] = hw.constant false
// CHECK:         %[[C1:.*]] = hw.constant true
// CHECK:         %[[E0:.*]] = hw.array_get %arr[%[[C0]]] : !hw.array<2x!esi.window<"TypeAwin1", !hw.struct<header1: i6, header2: i1, header3: !hw.array<13xi16>>, [<"FrameA", [<"header1">, <"header2">]>, <"FrameB", [<"header3", 3>]>]>>, i1
// CHECK:         %[[U0:.*]] = esi.window.unwrap %[[E0]] : !esi.window<"TypeAwin1", !hw.struct<header1: i6, header2: i1, header3: !hw.array<13xi16>>, [<"FrameA", [<"header1">, <"header2">]>, <"FrameB", [<"header3", 3>]>]>
// CHECK:         %[[W0:.*]] = esi.window.wrap %[[U0]] : !esi.window<"TypeAwin1", !hw.struct<header1: i6, header2: i1, header3: !hw.array<13xi16>>, [<"FrameA", [<"header1">, <"header2">]>, <"FrameB", [<"header3", 3>]>]>
// CHECK:         %[[E1:.*]] = hw.array_get %arr[%[[C1]]] : !hw.array<2x!esi.window<"TypeAwin1", !hw.struct<header1: i6, header2: i1, header3: !hw.array<13xi16>>, [<"FrameA", [<"header1">, <"header2">]>, <"FrameB", [<"header3", 3>]>]>>, i1
// CHECK:         %[[U1:.*]] = esi.window.unwrap %[[E1]] : !esi.window<"TypeAwin1", !hw.struct<header1: i6, header2: i1, header3: !hw.array<13xi16>>, [<"FrameA", [<"header1">, <"header2">]>, <"FrameB", [<"header3", 3>]>]>
// CHECK:         %[[W1:.*]] = esi.window.wrap %[[U1]] : !esi.window<"TypeAwin1", !hw.struct<header1: i6, header2: i1, header3: !hw.array<13xi16>>, [<"FrameA", [<"header1">, <"header2">]>, <"FrameB", [<"header3", 3>]>]>
// CHECK:         %[[ARR:.*]] = hw.array_create %[[W1]], %[[W0]] : !esi.window<"TypeAwin1", !hw.struct<header1: i6, header2: i1, header3: !hw.array<13xi16>>, [<"FrameA", [<"header1">, <"header2">]>, <"FrameB", [<"header3", 3>]>]>
// LOW-LABEL: hw.module @ArrayWindowModule(
// LOW-NEXT:    %[[F:.*]] = hw.constant false
// LOW-NEXT:    %[[T:.*]] = hw.constant true
// LOW-NEXT:    %[[E0:.*]] = hw.array_get %arr[%[[F]]] : !hw.array<2xunion<FrameA: !hw.struct<header1: i6, header2: i1>, FrameB: !hw.struct<header3: !hw.array<3xi16>>, FrameB_leftOver: !hw.struct<header3: !hw.array<1xi16>>>>, i1
// LOW-NEXT:    %[[E1:.*]] = hw.array_get %arr[%[[T]]] : !hw.array<2xunion<FrameA: !hw.struct<header1: i6, header2: i1>, FrameB: !hw.struct<header3: !hw.array<3xi16>>, FrameB_leftOver: !hw.struct<header3: !hw.array<1xi16>>>>, i1
// LOW-NEXT:    %[[NEW_ARR:.*]] = hw.array_create %[[E1]], %[[E0]] : !hw.union<FrameA: !hw.struct<header1: i6, header2: i1>, FrameB: !hw.struct<header3: !hw.array<3xi16>>, FrameB_leftOver: !hw.struct<header3: !hw.array<1xi16>>>
// LOW-NEXT:    hw.output %[[NEW_ARR]] : !hw.array<2xunion<FrameA: !hw.struct<header1: i6, header2: i1>, FrameB: !hw.struct<header3: !hw.array<3xi16>>, FrameB_leftOver: !hw.struct<header3: !hw.array<1xi16>>>>
hw.module @ArrayWindowModule(in %arr: !ArrayWindow, out arr_out: !ArrayWindow) {
  %c0 = hw.constant 0 : i1
  %c1 = hw.constant 1 : i1
  %elem0 = hw.array_get %arr[%c0] : !ArrayWindow, i1
  %unwrap0 = esi.window.unwrap %elem0 : !TypeAwin1
  %rewrap0 = esi.window.wrap %unwrap0 : !TypeAwin1
  %elem1 = hw.array_get %arr[%c1] : !ArrayWindow, i1
  %unwrap1 = esi.window.unwrap %elem1 : !TypeAwin1
  %rewrap1 = esi.window.wrap %unwrap1 : !TypeAwin1
  %newArr = hw.array_create %rewrap1, %rewrap0 : !TypeAwin1
  hw.output %newArr : !ArrayWindow
}

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

// Test that structs containing windows are lowered correctly.
!InnerWindowStruct = !hw.struct<intField: i332, winField: !TypeAwin1>
// LOW-LABEL: hw.module @StructWindowModule(
// LOW-SAME: in %a : !hw.struct<intField: i332, winField: !hw.union<FrameA: !hw.struct<header1: i6, header2: i1>, FrameB: !hw.struct<header3: !hw.array<3xi16>>, FrameB_leftOver: !hw.struct<header3: !hw.array<1xi16>>>
// LOW-SAME: out x : !hw.struct<intField: i332, winField: !hw.union<FrameA: !hw.struct<header1: i6, header2: i1>, FrameB: !hw.struct<header3: !hw.array<3xi16>>, FrameB_leftOver: !hw.struct<header3: !hw.array<1xi16>>>
// LOW-NEXT:  %intField = hw.struct_extract %a["intField"] : !hw.struct<intField: i332, winField: !hw.union<FrameA: !hw.struct<header1: i6, header2: i1>, FrameB: !hw.struct<header3: !hw.array<3xi16>>, FrameB_leftOver: !hw.struct<header3: !hw.array<1xi16>>>>
// LOW-NEXT:  %winField = hw.struct_extract %a["winField"] : !hw.struct<intField: i332, winField: !hw.union<FrameA: !hw.struct<header1: i6, header2: i1>, FrameB: !hw.struct<header3: !hw.array<3xi16>>, FrameB_leftOver: !hw.struct<header3: !hw.array<1xi16>>>>
// LOW-NEXT:  [[RET:%.+]] = hw.struct_create (%intField, %winField) : !hw.struct<intField: i332, winField: !hw.union<FrameA: !hw.struct<header1: i6, header2: i1>, FrameB: !hw.struct<header3: !hw.array<3xi16>>, FrameB_leftOver: !hw.struct<header3: !hw.array<1xi16>>>>
// LOW-NEXT:  hw.output [[RET]] : !hw.struct<intField: i332, winField: !hw.union<FrameA: !hw.struct<header1: i6, header2: i1>, FrameB: !hw.struct<header3: !hw.array<3xi16>>, FrameB_leftOver: !hw.struct<header3: !hw.array<1xi16>>>>
hw.module @StructWindowModule(in %a: !InnerWindowStruct, out x: !InnerWindowStruct) {
  %structIntField = hw.struct_extract %a[ "intField" ] : !InnerWindowStruct
  %win = hw.struct_extract %a[ "winField" ] : !InnerWindowStruct
  %unwrapped = esi.window.unwrap %win : !TypeAwin1
  %rewrapped = esi.window.wrap %unwrapped : !TypeAwin1
  %outStruct = hw.struct_create ( %structIntField, %rewrapped ) : !InnerWindowStruct
  hw.output %outStruct : !InnerWindowStruct
}

// CHECK-LABEL: hw.module @UnionWindowModule
// CHECK:         %[[W:.*]] = hw.union_extract %u["frame"] : !hw.union<frame: !esi.window<"TypeAwin1", !hw.struct<header1: i6, header2: i1, header3: !hw.array<13xi16>>, [<"FrameA", [<"header1">, <"header2">]>, <"FrameB", [<"header3", 3>]>]>>
// CHECK:         %[[U:.*]] = esi.window.unwrap %[[W]] : !esi.window<"TypeAwin1", !hw.struct<header1: i6, header2: i1, header3: !hw.array<13xi16>>, [<"FrameA", [<"header1">, <"header2">]>, <"FrameB", [<"header3", 3>]>]>
// CHECK:         %[[R:.*]] = esi.window.wrap %[[U]] : !esi.window<"TypeAwin1", !hw.struct<header1: i6, header2: i1, header3: !hw.array<13xi16>>, [<"FrameA", [<"header1">, <"header2">]>, <"FrameB", [<"header3", 3>]>]>
// CHECK:         %[[OUT:.*]] = hw.union_create "frame", %[[R]] : !hw.union<frame: !esi.window<"TypeAwin1", !hw.struct<header1: i6, header2: i1, header3: !hw.array<13xi16>>, [<"FrameA", [<"header1">, <"header2">]>, <"FrameB", [<"header3", 3>]>]>>
// LOW-LABEL: hw.module @UnionWindowModule(
// LOW-NEXT:    %[[EXTRACT:.*]] = hw.union_extract %u["frame"] : !hw.union<frame: !hw.union<FrameA: !hw.struct<header1: i6, header2: i1>, FrameB: !hw.struct<header3: !hw.array<3xi16>>, FrameB_leftOver: !hw.struct<header3: !hw.array<1xi16>>>>
// LOW-NEXT:    %[[REPACK:.*]] = hw.union_create "frame", %[[EXTRACT]] : !hw.union<frame: !hw.union<FrameA: !hw.struct<header1: i6, header2: i1>, FrameB: !hw.struct<header3: !hw.array<3xi16>>, FrameB_leftOver: !hw.struct<header3: !hw.array<1xi16>>>>
// LOW-NEXT:    hw.output %[[REPACK]] : !hw.union<frame: !hw.union<FrameA: !hw.struct<header1: i6, header2: i1>, FrameB: !hw.struct<header3: !hw.array<3xi16>>, FrameB_leftOver: !hw.struct<header3: !hw.array<1xi16>>>>
hw.module @UnionWindowModule(in %u: !hw.union<frame: !TypeAwin1>, out x: !hw.union<frame: !TypeAwin1>) {
  %win = hw.union_extract %u["frame"] : !hw.union<frame: !TypeAwin1>
  %unwrapped = esi.window.unwrap %win : !TypeAwin1
  %rewrapped = esi.window.wrap %unwrapped : !TypeAwin1
  %out = hw.union_create "frame", %rewrapped : !hw.union<frame: !TypeAwin1>
  hw.output %out : !hw.union<frame: !TypeAwin1>
}

// CHECK-LABEL: hw.module @AliasStructWindowModule
// CHECK:         %[[FIELD:.*]] = hw.struct_extract %a["aliasField"] : !hw.typealias<@AliasStruct, !hw.struct<aliasField: !esi.window<"TypeAwin1", !hw.struct<header1: i6, header2: i1, header3: !hw.array<13xi16>>, [<"FrameA", [<"header1">, <"header2">]>, <"FrameB", [<"header3", 3>]>]>>>
// CHECK:         %[[U:.*]] = esi.window.unwrap %[[FIELD]] : !esi.window<"TypeAwin1", !hw.struct<header1: i6, header2: i1, header3: !hw.array<13xi16>>, [<"FrameA", [<"header1">, <"header2">]>, <"FrameB", [<"header3", 3>]>]>
// CHECK:         %[[R:.*]] = esi.window.wrap %[[U]] : !esi.window<"TypeAwin1", !hw.struct<header1: i6, header2: i1, header3: !hw.array<13xi16>>, [<"FrameA", [<"header1">, <"header2">]>, <"FrameB", [<"header3", 3>]>]>
// CHECK:         %[[STRUCT:.*]] = hw.struct_create (%[[R]]) : !hw.typealias<@AliasStruct, !hw.struct<aliasField: !esi.window<"TypeAwin1", !hw.struct<header1: i6, header2: i1, header3: !hw.array<13xi16>>, [<"FrameA", [<"header1">, <"header2">]>, <"FrameB", [<"header3", 3>]>]>>>
// LOW-LABEL: hw.module @AliasStructWindowModule(
// LOW-NEXT:    %[[FIELD:.*]] = hw.struct_extract %a["aliasField"] : !hw.typealias<@AliasStruct, !hw.struct<aliasField: !hw.union<FrameA: !hw.struct<header1: i6, header2: i1>, FrameB: !hw.struct<header3: !hw.array<3xi16>>, FrameB_leftOver: !hw.struct<header3: !hw.array<1xi16>>>>>
// LOW-NEXT:    %[[STRUCT:.*]] = hw.struct_create (%[[FIELD]]) : !hw.typealias<@AliasStruct, !hw.struct<aliasField: !hw.union<FrameA: !hw.struct<header1: i6, header2: i1>, FrameB: !hw.struct<header3: !hw.array<3xi16>>, FrameB_leftOver: !hw.struct<header3: !hw.array<1xi16>>>>>
// LOW-NEXT:    hw.output %[[STRUCT]] : !hw.typealias<@AliasStruct, !hw.struct<aliasField: !hw.union<FrameA: !hw.struct<header1: i6, header2: i1>, FrameB: !hw.struct<header3: !hw.array<3xi16>>, FrameB_leftOver: !hw.struct<header3: !hw.array<1xi16>>>>>
hw.module @AliasStructWindowModule(in %a: !AliasStruct, out x: !AliasStruct) {
  %field = hw.struct_extract %a["aliasField"] : !AliasStruct
  %unwrapped = esi.window.unwrap %field : !TypeAwin1
  %rewrapped = esi.window.wrap %unwrapped : !TypeAwin1
  %struct = hw.struct_create (%rewrapped) : !AliasStruct
  hw.output %struct : !AliasStruct
}

// CHECK-LABEL: hw.module.extern @ListWindowModule(in %list : !esi.list<!esi.window<"TypeAwin1", !hw.struct<header1: i6, header2: i1, header3: !hw.array<13xi16>>, [<"FrameA", [<"header1">, <"header2">]>, <"FrameB", [<"header3", 3>]>]>>)
// LOW-LABEL: hw.module.extern @ListWindowModule(in %list : !esi.list<!hw.union<FrameA: !hw.struct<header1: i6, header2: i1>, FrameB: !hw.struct<header3: !hw.array<3xi16>>, FrameB_leftOver: !hw.struct<header3: !hw.array<1xi16>>>>)
hw.module.extern @ListWindowModule(in %list: !ListOfWindows)

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

// Test TypeAlias lowering with single unnamed frame
!SimpleStruct = !hw.typealias<@pycde::@SimpleStruct, !hw.struct<field1: i8, field2: i16>>
!SimpleWindow = !esi.window<"SimpleWin", !SimpleStruct, [
  <"", [<"field1">, <"field2">]>
]>

// The lowered type of a single unnamed frame window should be the struct itself.
// If the 'into' type is a TypeAlias, the lowered type should also be a TypeAlias
// named '<originalAlias>_<windowName>'.
// CHECK-LABEL: hw.module @SimpleWindowModule(in %a : !esi.window<"SimpleWin", !hw.typealias<@pycde::@SimpleStruct, !hw.struct<field1: i8, field2: i16>>, [<"", [<"field1">, <"field2">]>]>, out x : !hw.typealias<@pycde::@SimpleStruct_SimpleWin, !hw.struct<field1: i8, field2: i16>>)
// LOW-LABEL: hw.module @SimpleWindowModule(in %a : !hw.typealias<@pycde::@SimpleStruct_SimpleWin, !hw.struct<field1: i8, field2: i16>>, out x : !hw.typealias<@pycde::@SimpleStruct_SimpleWin, !hw.struct<field1: i8, field2: i16>>)
hw.module @SimpleWindowModule(in %a: !SimpleWindow, out x: !hw.typealias<@pycde::@SimpleStruct_SimpleWin, !hw.struct<field1: i8, field2: i16>>) {
  %u = esi.window.unwrap %a : !SimpleWindow
  hw.output %u : !hw.typealias<@pycde::@SimpleStruct_SimpleWin, !hw.struct<field1: i8, field2: i16>>
}

// Test TypeAlias lowering with multiple frames (non-special case)
!MultiFrameStruct = !hw.typealias<@pycde::@MultiFrameStruct, !hw.struct<header: i8, data: i16, tail: i8>>
!MultiFrameWindow = !esi.window<"MultiWin", !MultiFrameStruct, [
  <"HeaderFrame", [<"header">]>,
  <"DataFrame", [<"data">]>,
  <"TailFrame", [<"tail">]>
]>

// For multiple frames, the lowered type is a union of structs.
// If the 'into' type is a TypeAlias, the lowered type should also be a TypeAlias
// named '<originalAlias>_<windowName>'.
// CHECK-LABEL: hw.module @MultiFrameWindowModule(in %a : !esi.window<"MultiWin", !hw.typealias<@pycde::@MultiFrameStruct, !hw.struct<header: i8, data: i16, tail: i8>>, [<"HeaderFrame", [<"header">]>, <"DataFrame", [<"data">]>, <"TailFrame", [<"tail">]>]>, out x : !hw.typealias<@pycde::@MultiFrameStruct_MultiWin, !hw.union<HeaderFrame: !hw.struct<header: i8>, DataFrame: !hw.struct<data: i16>, TailFrame: !hw.struct<tail: i8>>>)
// LOW-LABEL: hw.module @MultiFrameWindowModule(in %a : !hw.typealias<@pycde::@MultiFrameStruct_MultiWin, !hw.union<HeaderFrame: !hw.struct<header: i8>, DataFrame: !hw.struct<data: i16>, TailFrame: !hw.struct<tail: i8>>>, out x : !hw.typealias<@pycde::@MultiFrameStruct_MultiWin, !hw.union<HeaderFrame: !hw.struct<header: i8>, DataFrame: !hw.struct<data: i16>, TailFrame: !hw.struct<tail: i8>>>)
hw.module @MultiFrameWindowModule(in %a: !MultiFrameWindow, out x: !hw.typealias<@pycde::@MultiFrameStruct_MultiWin, !hw.union<HeaderFrame: !hw.struct<header: i8>, DataFrame: !hw.struct<data: i16>, TailFrame: !hw.struct<tail: i8>>>) {
  %u = esi.window.unwrap %a : !MultiFrameWindow
  hw.output %u : !hw.typealias<@pycde::@MultiFrameStruct_MultiWin, !hw.union<HeaderFrame: !hw.struct<header: i8>, DataFrame: !hw.struct<data: i16>, TailFrame: !hw.struct<tail: i8>>>
}
