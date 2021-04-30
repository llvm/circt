// RUN: circt-opt -rtl-declare-typedefs %s | FileCheck %s

// CHECK-DAG: sv.typedef @foo : i1
// CHECK-DAG: sv.typedef @bar : !rtl.array<4xi32>
// CHECK-DAG: sv.typedef @baz : i128

sv.interface @iface {
  sv.interface.signal @data : !rtl.typealias<bar, array<4xi32>>
}

func @test(%arg0: !rtl.typealias<foo, i1>) -> !rtl.typealias<baz, i128> {
  %ifaceInst = sv.interface.instance : !sv.interface<@iface>
  %data = sv.interface.signal.read %ifaceInst(@iface::@data) : !rtl.typealias<bar, array<4xi32>>
  %garbage = rtl.bitcast %data : (!rtl.typealias<bar, array<4xi32>>) -> !rtl.typealias<baz, i128>
  return %garbage : !rtl.typealias<baz, i128>
}
