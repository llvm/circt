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
