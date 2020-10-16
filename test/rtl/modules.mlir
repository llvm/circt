// RUN: circt-opt %s -verify-diagnostics | circt-opt -verify-diagnostics | FileCheck %s

module {
  rtl.module @B(%a: i1) -> (i1 { rtl.name = "nameOfPortInSV"}, i1) {
    %0 = rtl.or %a, %a : i1
    %1 = rtl.and %a, %a : i1
    rtl.output %0, %1 : i1, i1
  }

  // CHECK-LABEL: rtl.module @B(%arg0: i1 {rtl.direction = "in", rtl.name = "a"}, %arg1: i1 {rtl.direction = "out", rtl.name = "b"}, %arg2: i1 {rtl.direction = "out", rtl.name = "c"})
  // CHECK-NEXT:    %0 = rtl.or %arg0, %arg0 : i1
  // CHECK-NEXT:    %1 = rtl.and %arg0, %arg0 : i1
  // CHECK-NEXT:    rtl.connect %arg1, %0 : i1
  // CHECK-NEXT:    rtl.connect %arg2, %1 : i1

  rtl.extmodule @C(%a: i1 {rtl.direction = "in", rtl.name = "nameOfPortInSV"}, 
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
                %f: i1 {rtl.direction = "out"},
                %g: i1 {rtl.direction = "out"}) {

    // Instantiate @B as a RTL module with result-as-output sementics
    %r1, %r2 = rtl.instance "b1" @B(%d) : (i1) -> (i1, i1)
    // Connect to an output port in the rtl.connect style
    rtl.connect %g, %r1 : i1
    // Instantiate @C the connect style
    rtl.instance "c1" @C(%d, %e, %f) : (i1, i1, i1) -> ()
  }
  // CHECK-LABEL: rtl.module @A(%arg0: i1 {rtl.direction = "in", rtl.name = "d"}, %arg1: i1 {rtl.direction = "in", rtl.name = "e"}, %arg2: i1 {rtl.direction = "out", rtl.name = "f"}) {
  // CHECK-NEXT:  rtl.instance "b1" @B(%arg0, %arg1, %arg2) : i1, i1, i1
  // CHECK-NEXT:  rtl.instance "c1" @C(%arg0, %arg1, %arg2) : i1, i1, i1

}
