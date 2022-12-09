// RUN: circt-opt -split-input-file -lower-handshake-to-firrtl %s | FileCheck %s

// CHECK-LABEL:   firrtl.circuit "multi_in"  {
// CHECK:           firrtl.module @handshake_sync_in_ui32_ui512_out_ui32_ui512(in %[[VAL_0:.*]]: !firrtl.bundle<valid: uint<1>, ready flip: uint<1>>, in %[[VAL_1:.*]]: !firrtl.bundle<valid: uint<1>, ready flip: uint<1>, data: uint<32>>, in %[[VAL_2:.*]]: !firrtl.bundle<valid: uint<1>, ready flip: uint<1>, data: uint<512>>, out %[[VAL_3:.*]]: !firrtl.bundle<valid: uint<1>, ready flip: uint<1>>, out %[[VAL_4:.*]]: !firrtl.bundle<valid: uint<1>, ready flip: uint<1>, data: uint<32>>, out %[[VAL_5:.*]]: !firrtl.bundle<valid: uint<1>, ready flip: uint<1>, data: uint<512>>, in %[[VAL_6:.*]]: !firrtl.clock, in %[[VAL_7:.*]]: !firrtl.uint<1>) {
// CHECK:             %[[VAL_8:.*]] = firrtl.subfield %[[VAL_0]][valid] : !firrtl.bundle<valid: uint<1>, ready flip: uint<1>>
// CHECK:             %[[VAL_9:.*]] = firrtl.subfield %[[VAL_0]][ready] : !firrtl.bundle<valid: uint<1>, ready flip: uint<1>>
// CHECK:             %[[VAL_10:.*]] = firrtl.subfield %[[VAL_1]][valid] : !firrtl.bundle<valid: uint<1>, ready flip: uint<1>, data: uint<32>>
// CHECK:             %[[VAL_11:.*]] = firrtl.subfield %[[VAL_1]][ready] : !firrtl.bundle<valid: uint<1>, ready flip: uint<1>, data: uint<32>>
// CHECK:             %[[VAL_12:.*]] = firrtl.subfield %[[VAL_1]][data] : !firrtl.bundle<valid: uint<1>, ready flip: uint<1>, data: uint<32>>
// CHECK:             %[[VAL_13:.*]] = firrtl.subfield %[[VAL_2]][valid] : !firrtl.bundle<valid: uint<1>, ready flip: uint<1>, data: uint<512>>
// CHECK:             %[[VAL_14:.*]] = firrtl.subfield %[[VAL_2]][ready] : !firrtl.bundle<valid: uint<1>, ready flip: uint<1>, data: uint<512>>
// CHECK:             %[[VAL_15:.*]] = firrtl.subfield %[[VAL_2]][data] : !firrtl.bundle<valid: uint<1>, ready flip: uint<1>, data: uint<512>>
// CHECK:             %[[VAL_16:.*]] = firrtl.subfield %[[VAL_3]][valid] : !firrtl.bundle<valid: uint<1>, ready flip: uint<1>>
// CHECK:             %[[VAL_17:.*]] = firrtl.subfield %[[VAL_3]][ready] : !firrtl.bundle<valid: uint<1>, ready flip: uint<1>>
// CHECK:             %[[VAL_18:.*]] = firrtl.subfield %[[VAL_4]][valid] : !firrtl.bundle<valid: uint<1>, ready flip: uint<1>, data: uint<32>>
// CHECK:             %[[VAL_19:.*]] = firrtl.subfield %[[VAL_4]][ready] : !firrtl.bundle<valid: uint<1>, ready flip: uint<1>, data: uint<32>>
// CHECK:             %[[VAL_20:.*]] = firrtl.subfield %[[VAL_4]][data] : !firrtl.bundle<valid: uint<1>, ready flip: uint<1>, data: uint<32>>
// CHECK:             %[[VAL_21:.*]] = firrtl.subfield %[[VAL_5]][valid] : !firrtl.bundle<valid: uint<1>, ready flip: uint<1>, data: uint<512>>
// CHECK:             %[[VAL_22:.*]] = firrtl.subfield %[[VAL_5]][ready] : !firrtl.bundle<valid: uint<1>, ready flip: uint<1>, data: uint<512>>
// CHECK:             %[[VAL_23:.*]] = firrtl.subfield %[[VAL_5]][data] : !firrtl.bundle<valid: uint<1>, ready flip: uint<1>, data: uint<512>>
// CHECK:             %[[VAL_24:.*]] = firrtl.wire   : !firrtl.uint<1>
// CHECK:             %[[VAL_25:.*]] = firrtl.wire   : !firrtl.uint<1>
// CHECK:             firrtl.connect %[[VAL_20]], %[[VAL_12]] : !firrtl.uint<32>, !firrtl.uint<32>
// CHECK:             firrtl.connect %[[VAL_23]], %[[VAL_15]] : !firrtl.uint<512>, !firrtl.uint<512>
// CHECK:             %[[VAL_26:.*]] = firrtl.and %[[VAL_10]], %[[VAL_8]] : (!firrtl.uint<1>, !firrtl.uint<1>) -> !firrtl.uint<1>
// CHECK:             %[[VAL_27:.*]] = firrtl.and %[[VAL_13]], %[[VAL_26]] : (!firrtl.uint<1>, !firrtl.uint<1>) -> !firrtl.uint<1>
// CHECK:             firrtl.connect %[[VAL_24]], %[[VAL_27]] : !firrtl.uint<1>, !firrtl.uint<1>
// CHECK:             %[[VAL_28:.*]] = firrtl.and %[[VAL_25]], %[[VAL_27]] : (!firrtl.uint<1>, !firrtl.uint<1>) -> !firrtl.uint<1>
// CHECK:             firrtl.connect %[[VAL_9]], %[[VAL_28]] : !firrtl.uint<1>, !firrtl.uint<1>
// CHECK:             firrtl.connect %[[VAL_11]], %[[VAL_28]] : !firrtl.uint<1>, !firrtl.uint<1>
// CHECK:             firrtl.connect %[[VAL_14]], %[[VAL_28]] : !firrtl.uint<1>, !firrtl.uint<1>
// CHECK:             %[[VAL_29:.*]] = firrtl.constant 0 : !firrtl.uint<1>
// CHECK:             %[[VAL_30:.*]] = firrtl.wire   : !firrtl.uint<1>
// CHECK:             %[[VAL_31:.*]] = firrtl.wire   : !firrtl.uint<1>
// CHECK:             %[[VAL_32:.*]] = firrtl.wire   : !firrtl.uint<1>
// CHECK:             %[[VAL_33:.*]] = firrtl.wire   : !firrtl.uint<1>
// CHECK:             %[[VAL_34:.*]] = firrtl.and %[[VAL_31]], %[[VAL_30]] : (!firrtl.uint<1>, !firrtl.uint<1>) -> !firrtl.uint<1>
// CHECK:             %[[VAL_35:.*]] = firrtl.and %[[VAL_32]], %[[VAL_34]] : (!firrtl.uint<1>, !firrtl.uint<1>) -> !firrtl.uint<1>
// CHECK:             firrtl.connect %[[VAL_33]], %[[VAL_35]] : !firrtl.uint<1>, !firrtl.uint<1>
// CHECK:             firrtl.connect %[[VAL_25]], %[[VAL_33]] : !firrtl.uint<1>, !firrtl.uint<1>
// CHECK:             %[[VAL_36:.*]] = firrtl.wire   : !firrtl.uint<1>
// CHECK:             %[[VAL_37:.*]] = firrtl.not %[[VAL_33]] : (!firrtl.uint<1>) -> !firrtl.uint<1>
// CHECK:             firrtl.connect %[[VAL_36]], %[[VAL_37]] : !firrtl.uint<1>, !firrtl.uint<1>
// CHECK:             %[[VAL_38:.*]] = firrtl.regreset  %[[VAL_6]], %[[VAL_7]], %[[VAL_29]]  : !firrtl.uint<1>, !firrtl.uint<1>, !firrtl.uint<1>
// CHECK:             %[[VAL_39:.*]] = firrtl.and %[[VAL_30]], %[[VAL_36]] : (!firrtl.uint<1>, !firrtl.uint<1>) -> !firrtl.uint<1>
// CHECK:             firrtl.connect %[[VAL_38]], %[[VAL_39]] : !firrtl.uint<1>, !firrtl.uint<1>
// CHECK:             %[[VAL_40:.*]] = firrtl.wire   : !firrtl.uint<1>
// CHECK:             %[[VAL_41:.*]] = firrtl.not %[[VAL_38]] : (!firrtl.uint<1>) -> !firrtl.uint<1>
// CHECK:             firrtl.connect %[[VAL_40]], %[[VAL_41]] : !firrtl.uint<1>, !firrtl.uint<1>
// CHECK:             %[[VAL_42:.*]] = firrtl.and %[[VAL_40]], %[[VAL_24]] : (!firrtl.uint<1>, !firrtl.uint<1>) -> !firrtl.uint<1>
// CHECK:             firrtl.connect %[[VAL_16]], %[[VAL_42]] : !firrtl.uint<1>, !firrtl.uint<1>
// CHECK:             %[[VAL_43:.*]] = firrtl.wire   : !firrtl.uint<1>
// CHECK:             %[[VAL_44:.*]] = firrtl.and %[[VAL_17]], %[[VAL_42]] : (!firrtl.uint<1>, !firrtl.uint<1>) -> !firrtl.uint<1>
// CHECK:             firrtl.connect %[[VAL_43]], %[[VAL_44]] : !firrtl.uint<1>, !firrtl.uint<1>
// CHECK:             %[[VAL_45:.*]] = firrtl.or %[[VAL_43]], %[[VAL_38]] : (!firrtl.uint<1>, !firrtl.uint<1>) -> !firrtl.uint<1>
// CHECK:             firrtl.connect %[[VAL_30]], %[[VAL_45]] : !firrtl.uint<1>, !firrtl.uint<1>
// CHECK:             %[[VAL_46:.*]] = firrtl.regreset  %[[VAL_6]], %[[VAL_7]], %[[VAL_29]]  : !firrtl.uint<1>, !firrtl.uint<1>, !firrtl.uint<1>
// CHECK:             %[[VAL_47:.*]] = firrtl.and %[[VAL_31]], %[[VAL_36]] : (!firrtl.uint<1>, !firrtl.uint<1>) -> !firrtl.uint<1>
// CHECK:             firrtl.connect %[[VAL_46]], %[[VAL_47]] : !firrtl.uint<1>, !firrtl.uint<1>
// CHECK:             %[[VAL_48:.*]] = firrtl.wire   : !firrtl.uint<1>
// CHECK:             %[[VAL_49:.*]] = firrtl.not %[[VAL_46]] : (!firrtl.uint<1>) -> !firrtl.uint<1>
// CHECK:             firrtl.connect %[[VAL_48]], %[[VAL_49]] : !firrtl.uint<1>, !firrtl.uint<1>
// CHECK:             %[[VAL_50:.*]] = firrtl.and %[[VAL_48]], %[[VAL_24]] : (!firrtl.uint<1>, !firrtl.uint<1>) -> !firrtl.uint<1>
// CHECK:             firrtl.connect %[[VAL_18]], %[[VAL_50]] : !firrtl.uint<1>, !firrtl.uint<1>
// CHECK:             %[[VAL_51:.*]] = firrtl.wire   : !firrtl.uint<1>
// CHECK:             %[[VAL_52:.*]] = firrtl.and %[[VAL_19]], %[[VAL_50]] : (!firrtl.uint<1>, !firrtl.uint<1>) -> !firrtl.uint<1>
// CHECK:             firrtl.connect %[[VAL_51]], %[[VAL_52]] : !firrtl.uint<1>, !firrtl.uint<1>
// CHECK:             %[[VAL_53:.*]] = firrtl.or %[[VAL_51]], %[[VAL_46]] : (!firrtl.uint<1>, !firrtl.uint<1>) -> !firrtl.uint<1>
// CHECK:             firrtl.connect %[[VAL_31]], %[[VAL_53]] : !firrtl.uint<1>, !firrtl.uint<1>
// CHECK:             %[[VAL_54:.*]] = firrtl.regreset  %[[VAL_6]], %[[VAL_7]], %[[VAL_29]]  : !firrtl.uint<1>, !firrtl.uint<1>, !firrtl.uint<1>
// CHECK:             %[[VAL_55:.*]] = firrtl.and %[[VAL_32]], %[[VAL_36]] : (!firrtl.uint<1>, !firrtl.uint<1>) -> !firrtl.uint<1>
// CHECK:             firrtl.connect %[[VAL_54]], %[[VAL_55]] : !firrtl.uint<1>, !firrtl.uint<1>
// CHECK:             %[[VAL_56:.*]] = firrtl.wire   : !firrtl.uint<1>
// CHECK:             %[[VAL_57:.*]] = firrtl.not %[[VAL_54]] : (!firrtl.uint<1>) -> !firrtl.uint<1>
// CHECK:             firrtl.connect %[[VAL_56]], %[[VAL_57]] : !firrtl.uint<1>, !firrtl.uint<1>
// CHECK:             %[[VAL_58:.*]] = firrtl.and %[[VAL_56]], %[[VAL_24]] : (!firrtl.uint<1>, !firrtl.uint<1>) -> !firrtl.uint<1>
// CHECK:             firrtl.connect %[[VAL_21]], %[[VAL_58]] : !firrtl.uint<1>, !firrtl.uint<1>
// CHECK:             %[[VAL_59:.*]] = firrtl.wire   : !firrtl.uint<1>
// CHECK:             %[[VAL_60:.*]] = firrtl.and %[[VAL_22]], %[[VAL_58]] : (!firrtl.uint<1>, !firrtl.uint<1>) -> !firrtl.uint<1>
// CHECK:             firrtl.connect %[[VAL_59]], %[[VAL_60]] : !firrtl.uint<1>, !firrtl.uint<1>
// CHECK:             %[[VAL_61:.*]] = firrtl.or %[[VAL_59]], %[[VAL_54]] : (!firrtl.uint<1>, !firrtl.uint<1>) -> !firrtl.uint<1>
// CHECK:             firrtl.connect %[[VAL_32]], %[[VAL_61]] : !firrtl.uint<1>, !firrtl.uint<1>
// CHECK:           }
// CHECK:           firrtl.module @multi_in(in %[[VAL_62:.*]]: !firrtl.bundle<valid: uint<1>, ready flip: uint<1>>, in %[[VAL_63:.*]]: !firrtl.bundle<valid: uint<1>, ready flip: uint<1>, data: uint<32>>, in %[[VAL_64:.*]]: !firrtl.bundle<valid: uint<1>, ready flip: uint<1>, data: uint<512>>, out %[[VAL_65:.*]]: !firrtl.bundle<valid: uint<1>, ready flip: uint<1>>, out %[[VAL_66:.*]]: !firrtl.bundle<valid: uint<1>, ready flip: uint<1>, data: uint<32>>, out %[[VAL_67:.*]]: !firrtl.bundle<valid: uint<1>, ready flip: uint<1>, data: uint<512>>, in %[[VAL_68:.*]]: !firrtl.clock, in %[[VAL_69:.*]]: !firrtl.uint<1>) {
// CHECK:             %[[VAL_70:.*]], %[[VAL_71:.*]], %[[VAL_72:.*]], %[[VAL_73:.*]], %[[VAL_74:.*]], %[[VAL_75:.*]], %[[VAL_76:.*]], %[[VAL_77:.*]] = firrtl.instance handshake_sync0  @handshake_sync_in_ui32_ui512_out_ui32_ui512(in in0: !firrtl.bundle<valid: uint<1>, ready flip: uint<1>>, in in1: !firrtl.bundle<valid: uint<1>, ready flip: uint<1>, data: uint<32>>, in in2: !firrtl.bundle<valid: uint<1>, ready flip: uint<1>, data: uint<512>>, out out0: !firrtl.bundle<valid: uint<1>, ready flip: uint<1>>, out out1: !firrtl.bundle<valid: uint<1>, ready flip: uint<1>, data: uint<32>>, out out2: !firrtl.bundle<valid: uint<1>, ready flip: uint<1>, data: uint<512>>, in clock: !firrtl.clock, in reset: !firrtl.uint<1>)
// CHECK:             firrtl.connect %[[VAL_70]], %[[VAL_62]] : !firrtl.bundle<valid: uint<1>, ready flip: uint<1>>, !firrtl.bundle<valid: uint<1>, ready flip: uint<1>>
// CHECK:             firrtl.connect %[[VAL_71]], %[[VAL_63]] : !firrtl.bundle<valid: uint<1>, ready flip: uint<1>, data: uint<32>>, !firrtl.bundle<valid: uint<1>, ready flip: uint<1>, data: uint<32>>
// CHECK:             firrtl.connect %[[VAL_72]], %[[VAL_64]] : !firrtl.bundle<valid: uint<1>, ready flip: uint<1>, data: uint<512>>, !firrtl.bundle<valid: uint<1>, ready flip: uint<1>, data: uint<512>>
// CHECK:             firrtl.connect %[[VAL_76]], %[[VAL_68]] : !firrtl.clock, !firrtl.clock
// CHECK:             firrtl.connect %[[VAL_77]], %[[VAL_69]] : !firrtl.uint<1>, !firrtl.uint<1>
// CHECK:             firrtl.connect %[[VAL_65]], %[[VAL_73]] : !firrtl.bundle<valid: uint<1>, ready flip: uint<1>>, !firrtl.bundle<valid: uint<1>, ready flip: uint<1>>
// CHECK:             firrtl.connect %[[VAL_66]], %[[VAL_74]] : !firrtl.bundle<valid: uint<1>, ready flip: uint<1>, data: uint<32>>, !firrtl.bundle<valid: uint<1>, ready flip: uint<1>, data: uint<32>>
// CHECK:             firrtl.connect %[[VAL_67]], %[[VAL_75]] : !firrtl.bundle<valid: uint<1>, ready flip: uint<1>, data: uint<512>>, !firrtl.bundle<valid: uint<1>, ready flip: uint<1>, data: uint<512>>
// CHECK:           }
// CHECK:         }

handshake.func @multi_in(%in0: none, %in1: i32, %in2: i512) -> (none, i32, i512) {
  %out:3 = sync %in0, %in1, %in2 : none, i32, i512
  return %out#0, %out#1, %out#2 : none, i32, i512
}
