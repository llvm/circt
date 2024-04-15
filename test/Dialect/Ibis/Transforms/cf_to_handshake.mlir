
// RUN: circt-opt --pass-pipeline="builtin.module(ibis.design(ibis.class(ibis-convert-cf-to-handshake)))" \
// RUN:      --allow-unregistered-dialect %s | FileCheck %s

// CHECK:           ibis.method.df @foo(%[[VAL_1:.*]]: i32, %[[VAL_2:.*]]: i32, %[[VAL_3:.*]]: i1, %[[VAL_4:.*]]: none) -> (i32, none) {
// CHECK:             %[[VAL_5:.*]] = handshake.merge %[[VAL_1]] : i32
// CHECK:             %[[VAL_6:.*]] = handshake.merge %[[VAL_2]] : i32
// CHECK:             %[[VAL_7:.*]] = handshake.merge %[[VAL_3]] : i1
// CHECK:             %[[VAL_8:.*]] = handshake.buffer [2] fifo %[[VAL_7]] : i1
// CHECK:             %[[VAL_9:.*]] = handshake.merge %[[VAL_4]] : none
// CHECK:             %[[VAL_10:.*]] = ibis.sblock (%[[VAL_11:.*]] : i32 = %[[VAL_5]], %[[VAL_12:.*]] : i32 = %[[VAL_6]]) -> i32 attributes {maxThreads = 1 : i64} {
// CHECK:               %[[VAL_13:.*]] = arith.addi %[[VAL_11]], %[[VAL_12]] : i32
// CHECK:               ibis.sblock.return %[[VAL_13]] : i32
// CHECK:             }
// CHECK:             %[[VAL_14:.*]], %[[VAL_15:.*]] = handshake.cond_br %[[VAL_7]], %[[VAL_5]] : i32
// CHECK:             %[[VAL_16:.*]], %[[VAL_17:.*]] = handshake.cond_br %[[VAL_7]], %[[VAL_9]] : none
// CHECK:             %[[VAL_18:.*]], %[[VAL_19:.*]] = handshake.cond_br %[[VAL_7]], %[[VAL_10]] : i32
// CHECK:             %[[VAL_20:.*]] = handshake.merge %[[VAL_14]] : i32
// CHECK:             %[[VAL_21:.*]] = handshake.merge %[[VAL_18]] : i32
// CHECK:             %[[VAL_22:.*]], %[[VAL_23:.*]] = handshake.control_merge %[[VAL_16]] : none, index
// CHECK:             %[[VAL_24:.*]] = ibis.sblock (%[[VAL_25:.*]] : i32 = %[[VAL_21]], %[[VAL_26:.*]] : i32 = %[[VAL_20]]) -> i32 {
// CHECK:               %[[VAL_27:.*]] = arith.subi %[[VAL_25]], %[[VAL_26]] : i32
// CHECK:               ibis.sblock.return %[[VAL_27]] : i32
// CHECK:             }
// CHECK:             %[[VAL_28:.*]] = handshake.br %[[VAL_22]] : none
// CHECK:             %[[VAL_29:.*]] = handshake.br %[[VAL_24]] : i32
// CHECK:             %[[VAL_30:.*]] = handshake.merge %[[VAL_15]] : i32
// CHECK:             %[[VAL_31:.*]] = handshake.merge %[[VAL_19]] : i32
// CHECK:             %[[VAL_32:.*]], %[[VAL_33:.*]] = handshake.control_merge %[[VAL_17]] : none, index
// CHECK:             %[[VAL_34:.*]] = ibis.sblock (%[[VAL_35:.*]] : i32 = %[[VAL_31]], %[[VAL_36:.*]] : i32 = %[[VAL_30]]) -> i32 {
// CHECK:               %[[VAL_37:.*]] = "foo.op2"(%[[VAL_35]], %[[VAL_36]]) : (i32, i32) -> i32
// CHECK:               ibis.sblock.return %[[VAL_37]] : i32
// CHECK:             }
// CHECK:             %[[VAL_38:.*]] = handshake.br %[[VAL_32]] : none
// CHECK:             %[[VAL_39:.*]] = handshake.br %[[VAL_34]] : i32
// CHECK:             %[[VAL_40:.*]] = handshake.mux %[[VAL_41:.*]] {{\[}}%[[VAL_39]], %[[VAL_29]]] : index, i32
// CHECK:             %[[VAL_42:.*]] = handshake.mux %[[VAL_8]] {{\[}}%[[VAL_38]], %[[VAL_28]]] : i1, none
// CHECK:             %[[VAL_41]] = arith.index_cast %[[VAL_8]] : i1 to index
// CHECK:             ibis.return %[[VAL_40]], %[[VAL_42]] : i32, none
// CHECK:           }

ibis.design @foo {
ibis.class @ToHandshake {
  %this = ibis.this <@ToHandshake>
  // Just a simple test demonstrating the intended mixing of `ibis.sblock`s and
  // control flow operations. The meat of cf-to-handshake conversion is tested
  // in the handshake dialect tests.
  ibis.method @foo(%a: i32, %b: i32, %cond : i1) -> i32 {
    %0 = ibis.sblock (%arg0 : i32 = %a, %arg1 : i32 = %b) -> i32 attributes {maxThreads = 1 : i64} {
      %4 = arith.addi %arg0, %arg1 : i32
      ibis.sblock.return %4 : i32
    }
    cf.cond_br %cond, ^bb1(%a, %0 : i32, i32), ^bb2(%a, %0 : i32, i32)
  ^bb1(%11: i32, %21: i32):  // pred: ^bb0
    %31 = ibis.sblock (%arg0 : i32 = %21, %arg1 : i32 = %11) -> i32 {
      %4 = arith.subi %arg0, %arg1 : i32
      ibis.sblock.return %4 : i32
    }
    cf.br ^bb4(%31 : i32)
  ^bb2(%12 : i32, %22 : i32):
    %32 = ibis.sblock (%arg0 : i32 = %22, %arg1 : i32 = %12) -> i32 {
      %4 = "foo.op2"(%arg0, %arg1) : (i32, i32) -> i32
      ibis.sblock.return %4 : i32
    }
    cf.br ^bb4(%32 : i32)
  ^bb4(%res : i32):
    ibis.return %res : i32
  }
}
}
