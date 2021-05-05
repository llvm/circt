// RUN: circt-opt -sv-extract-test-code %s | FileCheck %s

// CHECK-LABEL:  rtl.module @extract_cover(%arg0: i1, %arg1: i1, %arg2: i1) attributes {argNames = ["", "", ""], outputPath = "generated/covers/*  */"} {
// CHECK-NEXT:    sv.always posedge %arg2  {
// CHECK-NEXT:      sv.ifdef.procedural "FUN_AND_GAMES"  {
// CHECK-NEXT:        %0 = comb.and %arg0, %arg1 : i1
// CHECK-NEXT:        sv.if %0  {
// CHECK-NEXT:        } else  {
// CHECK-NEXT:          sv.cover %arg1 : i1
// CHECK-NEXT:        }
// CHECK-NEXT:      }
// CHECK-NEXT:    }
// CHECK-NEXT:    rtl.output
// CHECK-NEXT:  }
// CHECK-NEXT:  rtl.module @extract_assert(%arg0: i1, %arg1: i1, %arg2: i1, %arg3: i1, %arg4: i1) attributes {argNames = ["", "", "", "", ""], outputPath = "generated/asserts"} {
// CHECK-NEXT:    sv.always posedge %arg3  {
// CHECK-NEXT:      sv.ifdef.procedural "FUN_AND_GAMES"  {
// CHECK-NEXT:        %0 = comb.and %arg0, %arg1 : i1
// CHECK-NEXT:        sv.if %0  {
// CHECK-NEXT:          sv.fwrite "this cond is split"
// CHECK-NEXT:          sv.assert %arg0 : i1
// CHECK-NEXT:        } else  {
// CHECK-NEXT:        }
// CHECK-NEXT:      }
// CHECK-NEXT:      sv.if %arg4  {
// CHECK-NEXT:        sv.assume %arg2 : i1
// CHECK-NEXT:      }
// CHECK-NEXT:    }
// CHECK-NEXT:    rtl.output
// CHECK-NEXT:  }
// CHECK: rtl.module @extract
// CHECK:       rtl.instance "InvisibleBind_assert" @extract_assert(%a, %b, %c, %clock, %0) {genAsBind = true} : (i1, i1, i1, i1, i1) -> ()
// CHECK-NEXT:  rtl.instance "InvisibleBind_cover" @extract_cover(%a, %b, %clock) {genAsBind = true} : (i1, i1, i1) -> ()
// CHECK-NEXT:  rtl.output

rtl.module @extract(%clock: i1, %a: i1, %b: i1, %c: i1) {
  %r = sv.reg : !rtl.inout<i1>
  %d = sv.read_inout %r : !rtl.inout<i1>
  sv.always posedge %clock  {
     sv.ifdef.procedural "FUN_AND_GAMES" {
       %cond = comb.and %a, %b : i1
       sv.if %cond  {
         sv.fwrite "this cond is split"
         sv.assert %a : i1
       } else {
           sv.cover %b : i1
       }
     }
     sv.if %d {
         sv.assume %c : i1
     }
  }
}

