// RUN: circt-opt -hw-declare-typedecls %s | FileCheck %s

// CHECK-LABEL: hw.type_scope @myscope
// CHECK-DAG: hw.typedecl @foo : i1
// CHECK-DAG: hw.typedecl @bar : !hw.array<4xi32>
// CHECK-DAG: hw.typedecl @baz : i128

sv.interface @iface {
  sv.interface.signal @data : !hw.typealias<@myscope::@bar, !hw.array<4xi32>>
}

func @test(%arg0: !hw.typealias<@myscope::@foo, i1>) -> !hw.typealias<@myscope::@baz, i128> {
  %ifaceInst = sv.interface.instance : !sv.interface<@iface>
  %data = sv.interface.signal.read %ifaceInst(@iface::@data) : !hw.typealias<@myscope::@bar, !hw.array<4xi32>>
  %garbage = hw.bitcast %data : (!hw.typealias<@myscope::@bar, !hw.array<4xi32>>) -> !hw.typealias<@myscope::@baz, i128>
  return %garbage : !hw.typealias<@myscope::@baz, i128>
}
