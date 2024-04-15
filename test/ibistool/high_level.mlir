// RUN: ibistool --hi --post-ibis-ir %s | FileCheck %s

// CHECK-LABEL:   ibis.class @ToHandshake {
// CHECK:           %[[VAL_0:.*]] = ibis.this <@ToHandshake>
// CHECK:           ibis.method @foo(%[[VAL_1:.*]]: i32, %[[VAL_2:.*]]: i32, %[[VAL_3:.*]]: i1) -> i32 {
// CHECK:             %[[VAL_4:.*]]:3 = ibis.sblock.isolated () -> (i32, i32, i32) {
// CHECK:               %[[VAL_5:.*]] = arith.constant 2 : i32
// CHECK:               %[[VAL_6:.*]] = arith.constant 1 : i32
// CHECK:               %[[VAL_7:.*]] = arith.constant 0 : i32
// CHECK:               ibis.sblock.return %[[VAL_5]], %[[VAL_6]], %[[VAL_7]] : i32, i32, i32
// CHECK:             }
// CHECK:             cf.br ^bb1(%[[VAL_1]], %[[VAL_8:.*]]#2, %[[VAL_2]], %[[VAL_8]]#0, %[[VAL_8]]#1, %[[VAL_8]]#2 : i32, i32, i32, i32, i32, i32)
// CHECK:           ^bb1(%[[VAL_9:.*]]: i32, %[[VAL_10:.*]]: i32, %[[VAL_11:.*]]: i32, %[[VAL_12:.*]]: i32, %[[VAL_13:.*]]: i32, %[[VAL_14:.*]]: i32):
// CHECK:             %[[VAL_15:.*]] = ibis.sblock.isolated (%[[VAL_16:.*]] : i32 = %[[VAL_9]], %[[VAL_17:.*]] : i32 = %[[VAL_11]]) -> i1 {
// CHECK:               %[[VAL_18:.*]] = arith.cmpi slt, %[[VAL_16]], %[[VAL_17]] : i32
// CHECK:               ibis.sblock.return %[[VAL_18]] : i1
// CHECK:             }
// CHECK:             cf.cond_br %[[VAL_15]], ^bb2(%[[VAL_11]], %[[VAL_12]], %[[VAL_13]], %[[VAL_14]], %[[VAL_9]], %[[VAL_10]] : i32, i32, i32, i32, i32, i32), ^bb6(%[[VAL_10]] : i32)
// CHECK:           ^bb2(%[[VAL_19:.*]]: i32, %[[VAL_20:.*]]: i32, %[[VAL_21:.*]]: i32, %[[VAL_22:.*]]: i32, %[[VAL_23:.*]]: i32, %[[VAL_24:.*]]: i32):
// CHECK:             %[[VAL_25:.*]] = ibis.sblock.isolated (%[[VAL_26:.*]] : i32 = %[[VAL_24]], %[[VAL_27:.*]] : i32 = %[[VAL_20]], %[[VAL_28:.*]] : i32 = %[[VAL_22]]) -> i1 {
// CHECK:               %[[VAL_29:.*]] = arith.remsi %[[VAL_26]], %[[VAL_27]] : i32
// CHECK:               %[[VAL_30:.*]] = arith.cmpi eq, %[[VAL_29]], %[[VAL_28]] : i32
// CHECK:               ibis.sblock.return %[[VAL_30]] : i1
// CHECK:             }
// CHECK:             cf.cond_br %[[VAL_25]], ^bb3(%[[VAL_19]], %[[VAL_20]], %[[VAL_21]], %[[VAL_22]], %[[VAL_23]], %[[VAL_24]] : i32, i32, i32, i32, i32, i32), ^bb4(%[[VAL_19]], %[[VAL_20]], %[[VAL_21]], %[[VAL_22]], %[[VAL_23]], %[[VAL_24]] : i32, i32, i32, i32, i32, i32)
// CHECK:           ^bb3(%[[VAL_31:.*]]: i32, %[[VAL_32:.*]]: i32, %[[VAL_33:.*]]: i32, %[[VAL_34:.*]]: i32, %[[VAL_35:.*]]: i32, %[[VAL_36:.*]]: i32):
// CHECK:             %[[VAL_37:.*]] = ibis.sblock.isolated (%[[VAL_38:.*]] : i32 = %[[VAL_36]], %[[VAL_39:.*]] : i32 = %[[VAL_35]]) -> i32 {
// CHECK:               %[[VAL_40:.*]] = arith.addi %[[VAL_38]], %[[VAL_39]] : i32
// CHECK:               ibis.sblock.return %[[VAL_40]] : i32
// CHECK:             }
// CHECK:             cf.br ^bb5(%[[VAL_37]], %[[VAL_31]], %[[VAL_32]], %[[VAL_33]], %[[VAL_34]], %[[VAL_35]] : i32, i32, i32, i32, i32, i32)
// CHECK:           ^bb4(%[[VAL_41:.*]]: i32, %[[VAL_42:.*]]: i32, %[[VAL_43:.*]]: i32, %[[VAL_44:.*]]: i32, %[[VAL_45:.*]]: i32, %[[VAL_46:.*]]: i32):
// CHECK:             %[[VAL_47:.*]] = ibis.sblock.isolated (%[[VAL_48:.*]] : i32 = %[[VAL_46]], %[[VAL_49:.*]] : i32 = %[[VAL_45]]) -> i32 {
// CHECK:               %[[VAL_50:.*]] = arith.subi %[[VAL_48]], %[[VAL_49]] : i32
// CHECK:               ibis.sblock.return %[[VAL_50]] : i32
// CHECK:             }
// CHECK:             cf.br ^bb5(%[[VAL_47]], %[[VAL_41]], %[[VAL_42]], %[[VAL_43]], %[[VAL_44]], %[[VAL_45]] : i32, i32, i32, i32, i32, i32)
// CHECK:           ^bb5(%[[VAL_51:.*]]: i32, %[[VAL_52:.*]]: i32, %[[VAL_53:.*]]: i32, %[[VAL_54:.*]]: i32, %[[VAL_55:.*]]: i32, %[[VAL_56:.*]]: i32):
// CHECK:             %[[VAL_57:.*]] = ibis.sblock.isolated (%[[VAL_58:.*]] : i32 = %[[VAL_56]], %[[VAL_59:.*]] : i32 = %[[VAL_54]]) -> i32 {
// CHECK:               %[[VAL_60:.*]] = arith.addi %[[VAL_58]], %[[VAL_59]] : i32
// CHECK:               ibis.sblock.return %[[VAL_60]] : i32
// CHECK:             }
// CHECK:             cf.br ^bb1(%[[VAL_57]], %[[VAL_51]], %[[VAL_52]], %[[VAL_53]], %[[VAL_54]], %[[VAL_55]] : i32, i32, i32, i32, i32, i32)
// CHECK:           ^bb6(%[[VAL_61:.*]]: i32):
// CHECK:             ibis.return %[[VAL_61]] : i32
// CHECK:           }
// CHECK:         }

ibis.design @foo {

ibis.class @ToHandshake {
  %this = ibis.this <@ToHandshake>
  ibis.method @foo(%a: i32, %b: i32, %c: i1) -> i32 {
    %c2_i32 = arith.constant 2 : i32
    %c1 = arith.constant 1 : i32
    %c0_i32 = arith.constant 0 : i32
    %alloca = memref.alloca() : memref<i32>
    memref.store %c0_i32, %alloca[] : memref<i32>
    cf.br ^bb1(%a : i32)
  ^bb1(%0: i32):  // 2 preds: ^bb0, ^bb6
    %1 = arith.cmpi slt, %0, %b : i32
    cf.cond_br %1, ^bb2, ^bb7
  ^bb2:  // pred: ^bb1
    %2 = memref.load %alloca[] : memref<i32>
    %4 = arith.remsi %2, %c2_i32 : i32
    %5 = arith.cmpi eq, %4, %c0_i32 : i32
    cf.cond_br %5, ^bb3, ^bb4
  ^bb3:  // pred: ^bb2
    %6 = arith.addi %2, %0 : i32
    cf.br ^bb5(%6 : i32)
  ^bb4:  // pred: ^bb2
    %7 = arith.subi %2, %0 : i32
    cf.br ^bb5(%7 : i32)
  ^bb5(%8: i32):  // 2 preds: ^bb3, ^bb4
    cf.br ^bb6
  ^bb6:  // pred: ^bb5
    memref.store %8, %alloca[] : memref<i32>
    %9 = arith.addi %0, %c1 : i32
    cf.br ^bb1(%9 : i32)
  ^bb7:  // pred: ^bb1
    %10 = memref.load %alloca[] : memref<i32>
    ibis.return %10 : i32
  }
}
}
