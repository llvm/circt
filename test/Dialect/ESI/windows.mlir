// RUN: circt-opt %s -verify-diagnostics | circt-opt -verify-diagnostics | FileCheck %s

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


// CHECK-LABEL:   hw.module.extern @TypeAModuleDst(%windowed: !esi.window<"TypeAwin1", !hw.struct<header1: i6, header2: i1, header3: !hw.array<13xi16>>, [<"FrameA", [<"header1">, <"header2">]>, <"FrameB", [<"header3", 3>]>]>)
hw.module.extern @TypeAModuleDst(%windowed: !TypeAwin1)
// CHECK-LABEL:   hw.module.extern @TypeAModuleSrc() -> (windowed: !esi.window<"TypeAwin1", !hw.struct<header1: i6, header2: i1, header3: !hw.array<13xi16>>, [<"FrameA", [<"header1">, <"header2">]>, <"FrameB", [<"header3", 3>]>]>)
hw.module.extern @TypeAModuleSrc() -> (windowed: !TypeAwin1)

!lowered = !hw.union<
  FrameA: !hw.struct<header1: i6, header2: i1>,
  FrameB: !hw.struct<header3: !hw.array<3xi16>>,
  FrameB_leftOver: !hw.struct<header3: !hw.array<1xi16>>>

// CHECK-LABEL: hw.module @TypeAModuleUnwrap
// CHECK:         [[r0:%.+]] = esi.window.unwrap %a : !esi.window<"TypeAwin1", !hw.struct<header1: i6, header2: i1, header3: !hw.array<13xi16>>, [<"FrameA", [<"header1">, <"header2">]>, <"FrameB", [<"header3", 3>]>]>
// CHECK          hw.output [[r0]] : !hw.union<FrameA: !hw.struct<header1: i6, header2: i1>, FrameB: !hw.struct<header3: !hw.array<3xi16>>, FrameB_leftOver: !hw.struct<header3: !hw.array<1xi16>>>
hw.module @TypeAModuleUnwrap(%a: !TypeAwin1) -> (x: !lowered) {
  %u = esi.window.unwrap %a : !TypeAwin1
  hw.output %u : !lowered
}


// CHECK-LABEL: hw.module @TypeAModuleWrapUnwrap
// CHECK:         [[r0:%.+]] = esi.window.unwrap %a : !esi.window<"TypeAwin1", !hw.struct<header1: i6, header2: i1, header3: !hw.array<13xi16>>, [<"FrameA", [<"header1">, <"header2">]>, <"FrameB", [<"header3", 3>]>]>
// CHECK:         [[r1:%.+]] = esi.window.wrap [[r0]] : !esi.window<"TypeAwin1", !hw.struct<header1: i6, header2: i1, header3: !hw.array<13xi16>>, [<"FrameA", [<"header1">, <"header2">]>, <"FrameB", [<"header3", 3>]>]>
hw.module @TypeAModuleWrapUnwrap(%a: !TypeAwin1) -> (x: !TypeAwin1) {
  %u = esi.window.unwrap %a : !TypeAwin1
  %x = esi.window.wrap %u : !TypeAwin1
  hw.output %x : !TypeAwin1
}
