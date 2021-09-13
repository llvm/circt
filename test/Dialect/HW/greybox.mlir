// RUN: circt-opt -hw-stub-external-modules %s | FileCheck %s

// CHECK-LABEL: hw.module @local(%arg0: !hw.inout<i1>, %arg1: i1) -> (%r: i3) {
// CHECK-NEXT:    %foo.r = hw.instance "foo" @remote(arg0: %arg0: !hw.inout<i1>, arg1: %arg1: i1) -> (r: i3)
// CHECK-NEXT:    hw.output %foo.r : i3
// CHECK-NEXT:  }
// CHECK-LABEL: hw.module @remote(%arg0: !hw.inout<i1>, %arg1: i1) -> (%r: i3) {
// CHECK-NEXT: %x_i3 = sv.constantX : i3
// CHECK-NEXT:   hw.output %x_i3 : i3
// CHECK-NEXT: }

hw.module.extern @remote(%arg0: !hw.inout<i1>, %arg1: i1) -> (%r : i3)

hw.module @local(%arg0: !hw.inout<i1>, %arg1: i1) -> (%r : i3) {
    %tr = hw.instance "foo" @remote(arg0: %arg0: !hw.inout<i1>, arg1: %arg1: i1) -> (r: i3) {parameters = {DEFAULT = 0 : i64}}
    hw.output %tr : i3
}
