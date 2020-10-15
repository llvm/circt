// RUN: circt-opt %s -verify-diagnostics | circt-opt -verify-diagnostics | FileCheck %s

module {
  rtl.module @B(%a: i1 {rtl.direction = "in"}, 
                %b: i1 {rtl.direction = "out"}, 
                %c: i1 {rtl.direction = "out"}) {
    %0 = rtl.or %a, %a : i1
    %1 = rtl.and %a, %a : i1
    rtl.connect %b, %0 : i1
    rtl.connect %c, %1 : i1
  }

  // CHECK-LABEL: rtl.module @B(%arg0: i1 {rtl.direction = "in", rtl.name = "a"}, %arg1: i1 {rtl.direction = "out", rtl.name = "b"}, %arg2: i1 {rtl.direction = "out", rtl.name = "c"})
  // CHECK-NEXT:    %0 = rtl.or %arg0, %arg0 : i1
  // CHECK-NEXT:    %1 = rtl.and %arg0, %arg0 : i1
  // CHECK-NEXT:    rtl.connect %arg1, %0 : i1
  // CHECK-NEXT:    rtl.connect %arg2, %1 : i1

  rtl.extmodule @C(%a: i1 {rtl.direction = "in"}, 
                   %b: i1 {rtl.direction = "out"}, 
                   %c: i1 {rtl.direction = "out"})

  // CHECK-LABEL: rtl.extmodule @C(i1 {rtl.direction = "in", rtl.name = "a"}, i1 {rtl.direction = "out", rtl.name = "b"}, i1 {rtl.direction = "out", rtl.name = "c"})
  // CHECK-NOT: {

  rtl.extmodule @D_ATTR(%a: i1 {rtl.direction = "in"}, 
                   %b: i1 {rtl.direction = "out"}, 
                   %c: i1 {rtl.direction = "out"}) attributes {filename = "test.v", parameters = {DEFAULT = 0 : i64}}

  // CHECK-LABEL: rtl.extmodule @D_ATTR(i1 {rtl.direction = "in", rtl.name = "a"}, i1 {rtl.direction = "out", rtl.name = "b"}, i1 {rtl.direction = "out", rtl.name = "c"}) attributes {filename = "test.v", parameters = {DEFAULT = 0 : i64}}
  // CHECK-NOT: {

  rtl.module @A(%d: i1 {rtl.direction = "in"}, 
                %e: i1 {rtl.direction = "in"}, 
                %f: i1 {rtl.direction = "out"}) {

    rtl.instance "b1" @B(%d, %e, %f) : i1, i1, i1
    rtl.instance "c1" @C(%d, %e, %f) : i1, i1, i1
  }
  // CHECK-LABEL: rtl.module @A(%arg0: i1 {rtl.direction = "in", rtl.name = "d"}, %arg1: i1 {rtl.direction = "in", rtl.name = "e"}, %arg2: i1 {rtl.direction = "out", rtl.name = "f"}) {
  // CHECK-NEXT:  rtl.instance "b1" @B(%arg0, %arg1, %arg2) : i1, i1, i1
  // CHECK-NEXT:  rtl.instance "c1" @C(%arg0, %arg1, %arg2) : i1, i1, i1

}
