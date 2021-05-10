// RUN: circt-opt -rtl-stub-external-modules %s | FileCheck %s

//CHECK-LABEL: rtl.module @local(%arg0: !rtl.inout<i1>, %arg1: i1) -> (%r: i3) {
//CHECK-NEXT:    %foo.r = rtl.instance "foo" @remote(%arg0, %arg1) : (!rtl.inout<i1>, i1) -> i3
//CHECK-NEXT:    rtl.output %foo.r : i3
//CHECK-NEXT:  }
//CHECK-LABEL: rtl.module @remote(%arg0: !rtl.inout<i1>, %arg1: i1) -> (%r: i3) {
//CHECK-NEXT: %x_i3 = sv.constantX : i3 
//CHECK-NEXT:   rtl.output %x_i3 : i3
//CHECK-NEXT: }
rtl.module.extern @remote(%arg0: !rtl.inout<i1>, %arg1: i1) -> (%r : i3)

rtl.module @local(%arg0: !rtl.inout<i1>, %arg1: i1) -> (%r : i3) {
    %tr = rtl.instance "foo" @remote(%arg0, %arg1) {parameters = {DEFAULT = 0 : i64}} : (!rtl.inout<i1>, i1) -> (i3)
    rtl.output %tr : i3
}
