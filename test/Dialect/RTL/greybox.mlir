// RUN: circt-opt -rtl-stub-external-modules %s | FileCheck %s

//CHECK-LABEL: rtl.module @remote(%arg0: i1, %arg1: i1) -> (%r: i3) {
//CHECK-NEXT: %x_i3 = sv.constantX : i3 
//CHECK-NEXT:   rtl.output %x_i3 : i3
//CHECK-NEXT: }

rtl.module.extern @remote(%arg0: i1, %arg1: i1) -> (%r : i3)
