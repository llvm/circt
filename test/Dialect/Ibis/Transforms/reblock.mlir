// RUN: circt-opt --pass-pipeline='builtin.module(ibis.class(ibis.method(ibis-reblock)))' %s | FileCheck %s

// CHECK: #[[$ATTR_0:.+]] = loc("dummy":0:0)

// CHECK-LABEL:   ibis.class @Argify {
// CHECK:           %[[VAL_0:.*]] = ibis.this @Argify
// CHECK:           ibis.method @bar(%[[VAL_1:.*]]: i32) -> i32 attributes {ibis.blockinfo = {"0" = {loc = #[[$ATTR_0]]}}} {
// CHECK:             ibis.sblock (){
// CHECK:               ibis.sblock.return
// CHECK:             }
// CHECK:             ibis.return %[[VAL_1]] : i32
// CHECK:           }
// CHECK:           ibis.method @foo(%[[VAL_2:.*]]: i32, %[[VAL_3:.*]]: i32) -> i32 attributes {ibis.blockinfo = {"0" = {loc = #[[$ATTR_0]]}, "1" = {loc = #[[$ATTR_0]]}, "2" = {loc = #[[$ATTR_0]]}}} {
// CHECK:             %[[VAL_4:.*]] = ibis.sblock () -> i32{
// CHECK:               %[[VAL_5:.*]] = arith.addi %[[VAL_2]], %[[VAL_3]] : i32
// CHECK:               ibis.sblock.return %[[VAL_5]] : i32
// CHECK:             }
// CHECK:             cf.br ^bb1(%[[VAL_2]], %[[VAL_4]] : i32, i32)
// CHECK:           ^bb1(%[[VAL_6:.*]]: i32, %[[VAL_7:.*]]: i32):
// CHECK:             %[[VAL_8:.*]] = ibis.call @bar(%[[VAL_6]]) : (i32) -> i32
// CHECK:             cf.br ^bb2(%[[VAL_6]], %[[VAL_8]] : i32, i32)
// CHECK:           ^bb2(%[[VAL_9:.*]]: i32, %[[VAL_10:.*]]: i32):
// CHECK:             %[[VAL_11:.*]] = ibis.sblock () -> i32{
// CHECK:               %[[VAL_12:.*]] = arith.addi %[[VAL_10]], %[[VAL_9]] : i32
// CHECK:               ibis.sblock.return %[[VAL_12]] : i32
// CHECK:             }
// CHECK:             ibis.return %[[VAL_11]] : i32
// CHECK:           }
// CHECK:         }

ibis.class @Argify {
  %this = ibis.this @Argify

  ibis.method @bar(%arg0 : i32) -> i32 attributes {
    "ibis.blockinfo" = {
      "0" = {
        "loc" = loc("dummy":0:0)
      }
    }
  } {
    ibis.return %arg0 : i32
  }

  ibis.method @foo(%arg0 : i32, %arg1 : i32) -> i32 attributes {
    "ibis.blockinfo" = {
      "0" = {
        "loc" = loc("dummy":0:0)
      },
      "1" = {
        "loc" = loc("dummy":0:0)
      },
      "2" = {
        "loc" = loc("dummy":0:0)
      }
    }
  } {
      %0 = arith.addi %arg0, %arg1 : i32
      cf.br ^bb1(%arg0, %0 : i32, i32)
    ^bb1(%aa : i32, %b : i32):
      %c = ibis.call @bar(%aa) : (i32) -> (i32)
      cf.br ^bb2(%aa, %c : i32, i32)
    ^bb2(%aaa : i32, %cc : i32):
      %d = arith.addi %cc, %aaa : i32
      ibis.return %d : i32
  }
}
