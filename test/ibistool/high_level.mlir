// RUN: ibistool --hi --post-ibis-ir %s | FileCheck %s

// CHECK-LABEL:   ibis.class @ToHandshake {
// CHECK:           %[[VAL_0:.*]] = ibis.this @ToHandshake
// CHECK:           ibis.method @foo(%[[VAL_1:.*]]: index, %[[VAL_2:.*]]: index, %[[VAL_3:.*]]: i1) -> i32 {
// CHECK:             %[[VAL_4:.*]]:3 = ibis.sblock.isolated () -> (i32, index, i32) {
// CHECK:               %[[VAL_5:.*]] = arith.constant 2 : i32
// CHECK:               %[[VAL_6:.*]] = arith.constant 1 : index
// CHECK:               %[[VAL_7:.*]] = arith.constant 0 : i32
// CHECK:               ibis.sblock.return %[[VAL_5]], %[[VAL_6]], %[[VAL_7]] : i32, index, i32
// CHECK:             }
// CHECK:             cf.br ^bb1(%[[VAL_1]], %[[VAL_8:.*]]#2, %[[VAL_2]], %[[VAL_8]]#0, %[[VAL_8]]#1, %[[VAL_8]]#2 : index, i32, index, i32, index, i32)
// CHECK:           ^bb1(%[[VAL_9:.*]]: index, %[[VAL_10:.*]]: i32, %[[VAL_11:.*]]: index, %[[VAL_12:.*]]: i32, %[[VAL_13:.*]]: index, %[[VAL_14:.*]]: i32):
// CHECK:             %[[VAL_15:.*]] = ibis.sblock.isolated (%[[VAL_16:.*]] : index = %[[VAL_9]], %[[VAL_17:.*]] : index = %[[VAL_11]]) -> i1 {
// CHECK:               %[[VAL_18:.*]] = arith.cmpi slt, %[[VAL_16]], %[[VAL_17]] : index
// CHECK:               ibis.sblock.return %[[VAL_18]] : i1
// CHECK:             }
// CHECK:             cf.cond_br %[[VAL_15]], ^bb2(%[[VAL_11]], %[[VAL_12]], %[[VAL_13]], %[[VAL_14]], %[[VAL_9]], %[[VAL_10]] : index, i32, index, i32, index, i32), ^bb6(%[[VAL_10]] : i32)
// CHECK:           ^bb2(%[[VAL_19:.*]]: index, %[[VAL_20:.*]]: i32, %[[VAL_21:.*]]: index, %[[VAL_22:.*]]: i32, %[[VAL_23:.*]]: index, %[[VAL_24:.*]]: i32):
// CHECK:             %[[VAL_25:.*]]:2 = ibis.sblock.isolated (%[[VAL_26:.*]] : index = %[[VAL_23]], %[[VAL_27:.*]] : i32 = %[[VAL_24]], %[[VAL_28:.*]] : i32 = %[[VAL_20]], %[[VAL_29:.*]] : i32 = %[[VAL_22]]) -> (i32, i1) {
// CHECK:               %[[VAL_30:.*]] = arith.index_cast %[[VAL_26]] : index to i32
// CHECK:               %[[VAL_31:.*]] = arith.remsi %[[VAL_27]], %[[VAL_28]] : i32
// CHECK:               %[[VAL_32:.*]] = arith.cmpi eq, %[[VAL_31]], %[[VAL_29]] : i32
// CHECK:               ibis.sblock.return %[[VAL_30]], %[[VAL_32]] : i32, i1
// CHECK:             }
// CHECK:             cf.cond_br %[[VAL_33:.*]]#1, ^bb3(%[[VAL_19]], %[[VAL_20]], %[[VAL_21]], %[[VAL_22]], %[[VAL_23]], %[[VAL_24]], %[[VAL_33]]#0 : index, i32, index, i32, index, i32, i32), ^bb4(%[[VAL_19]], %[[VAL_20]], %[[VAL_21]], %[[VAL_22]], %[[VAL_23]], %[[VAL_24]], %[[VAL_33]]#0 : index, i32, index, i32, index, i32, i32)
// CHECK:           ^bb3(%[[VAL_34:.*]]: index, %[[VAL_35:.*]]: i32, %[[VAL_36:.*]]: index, %[[VAL_37:.*]]: i32, %[[VAL_38:.*]]: index, %[[VAL_39:.*]]: i32, %[[VAL_40:.*]]: i32):
// CHECK:             %[[VAL_41:.*]] = ibis.sblock.isolated (%[[VAL_42:.*]] : i32 = %[[VAL_39]], %[[VAL_43:.*]] : i32 = %[[VAL_40]]) -> i32 {
// CHECK:               %[[VAL_44:.*]] = arith.addi %[[VAL_42]], %[[VAL_43]] : i32
// CHECK:               ibis.sblock.return %[[VAL_44]] : i32
// CHECK:             }
// CHECK:             cf.br ^bb5(%[[VAL_41]], %[[VAL_34]], %[[VAL_35]], %[[VAL_36]], %[[VAL_37]], %[[VAL_38]] : i32, index, i32, index, i32, index)
// CHECK:           ^bb4(%[[VAL_45:.*]]: index, %[[VAL_46:.*]]: i32, %[[VAL_47:.*]]: index, %[[VAL_48:.*]]: i32, %[[VAL_49:.*]]: index, %[[VAL_50:.*]]: i32, %[[VAL_51:.*]]: i32):
// CHECK:             %[[VAL_52:.*]] = ibis.sblock.isolated (%[[VAL_53:.*]] : i32 = %[[VAL_50]], %[[VAL_54:.*]] : i32 = %[[VAL_51]]) -> i32 {
// CHECK:               %[[VAL_55:.*]] = arith.subi %[[VAL_53]], %[[VAL_54]] : i32
// CHECK:               ibis.sblock.return %[[VAL_55]] : i32
// CHECK:             }
// CHECK:             cf.br ^bb5(%[[VAL_52]], %[[VAL_45]], %[[VAL_46]], %[[VAL_47]], %[[VAL_48]], %[[VAL_49]] : i32, index, i32, index, i32, index)
// CHECK:           ^bb5(%[[VAL_56:.*]]: i32, %[[VAL_57:.*]]: index, %[[VAL_58:.*]]: i32, %[[VAL_59:.*]]: index, %[[VAL_60:.*]]: i32, %[[VAL_61:.*]]: index):
// CHECK:             %[[VAL_62:.*]] = ibis.sblock.isolated (%[[VAL_63:.*]] : index = %[[VAL_61]], %[[VAL_64:.*]] : index = %[[VAL_59]]) -> index {
// CHECK:               %[[VAL_65:.*]] = arith.addi %[[VAL_63]], %[[VAL_64]] : index
// CHECK:               ibis.sblock.return %[[VAL_65]] : index
// CHECK:             }
// CHECK:             cf.br ^bb1(%[[VAL_62]], %[[VAL_56]], %[[VAL_57]], %[[VAL_58]], %[[VAL_59]], %[[VAL_60]] : index, i32, index, i32, index, i32)
// CHECK:           ^bb6(%[[VAL_66:.*]]: i32):
// CHECK:             ibis.return %[[VAL_66]] : i32
// CHECK:           }
// CHECK:         }

ibis.class @ToHandshake {
  %this = ibis.this @ToHandshake
  ibis.method @foo(%a: index, %b: index, %c : i1) -> i32 {
    %sum = memref.alloca () : memref<i32>
    %c0_i32 = arith.constant 0 : i32
    memref.store %c0_i32, %sum[] : memref<i32>
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : i32
    %c0 = arith.constant 0 : i32
    scf.for %i = %a to %b step %c1 {
      %acc = memref.load %sum[] : memref<i32>
      %i_i32 = arith.index_cast %i : index to i32
            %rem = arith.remsi %acc, %c2 : i32
      %cond = arith.cmpi eq, %rem, %c0 : i32
      %res = scf.if %cond -> (i32) {
        %v = arith.addi %acc, %i_i32 : i32
        scf.yield %v : i32
      } else {
        %v = arith.subi %acc, %i_i32 : i32
        scf.yield %v : i32
      }
      memref.store %res, %sum[] : memref<i32>
    }
    %res = memref.load %sum[] : memref<i32>
    ibis.return %res : i32
  }
}
