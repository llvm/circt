// RUN: circt-opt -state-struct-generate=emit-none-info %s | FileCheck %s
// This test checks whether the state struct is generated correctly, and overrides module correctly. 

module {
// CHECK-LABEL: llvm.mlir.global internal @_Struct_always_inline_expr() {addr_space = 0 : i32} : !llvm.struct<"_Struct_always_inline_expr", packed (i1, i1, i1, i1, i1, i1, i1, i5, i5, array<2 x i5>, i1, array<2 x i5>)> {
// CHECK: %[[VAL_0:.*]] = llvm.mlir.constant(false) : i1
// CHECK: %[[VAL_1:.*]] = llvm.mlir.constant(false) : i1
// CHECK: %[[VAL_2:.*]] = llvm.mlir.constant(false) : i1
// CHECK: %[[VAL_3:.*]] = llvm.mlir.constant(false) : i1
// CHECK: %[[VAL_4:.*]] = llvm.mlir.constant(false) : i1
// CHECK: %[[VAL_5:.*]] = llvm.mlir.constant(false) : i1
// CHECK: %[[VAL_6:.*]] = llvm.mlir.constant(false) : i1
// CHECK: %[[VAL_7:.*]] = llvm.mlir.constant(false) : i1
// CHECK: %[[VAL_8:.*]] = llvm.mlir.constant(0 : i5) : i5
// CHECK: %[[VAL_9:.*]] = llvm.mlir.constant(0 : i5) : i5
// CHECK: %[[VAL_10:.*]] = llvm.mlir.constant(0 : i5) : i5
// CHECK: %[[VAL_11:.*]] = llvm.mlir.undef : !llvm.array<2 x i5>
// CHECK: %[[VAL_12:.*]] = llvm.insertvalue %[[VAL_10]], %[[VAL_11]][0] : !llvm.array<2 x i5>
// CHECK: %[[VAL_13:.*]] = llvm.insertvalue %[[VAL_10]], %[[VAL_12]][1] : !llvm.array<2 x i5>
// CHECK: %[[VAL_14:.*]] = llvm.mlir.undef : !llvm.struct<"_Struct_always_inline_expr", packed (i1, i1, i1, i1, i1, i1, i1, i5, i5, array<2 x i5>, i1, array<2 x i5>)>
// CHECK: %[[VAL_15:.*]] = llvm.insertvalue %[[VAL_0]], %[[VAL_14]][0] : !llvm.struct<"_Struct_always_inline_expr", packed (i1, i1, i1, i1, i1, i1, i1, i5, i5, array<2 x i5>, i1, array<2 x i5>)>
// CHECK: %[[VAL_16:.*]] = llvm.insertvalue %[[VAL_1]], %[[VAL_15]][1] : !llvm.struct<"_Struct_always_inline_expr", packed (i1, i1, i1, i1, i1, i1, i1, i5, i5, array<2 x i5>, i1, array<2 x i5>)>
// CHECK: %[[VAL_17:.*]] = llvm.insertvalue %[[VAL_2]], %[[VAL_16]][2] : !llvm.struct<"_Struct_always_inline_expr", packed (i1, i1, i1, i1, i1, i1, i1, i5, i5, array<2 x i5>, i1, array<2 x i5>)>
// CHECK: %[[VAL_18:.*]] = llvm.insertvalue %[[VAL_3]], %[[VAL_17]][3] : !llvm.struct<"_Struct_always_inline_expr", packed (i1, i1, i1, i1, i1, i1, i1, i5, i5, array<2 x i5>, i1, array<2 x i5>)>
// CHECK: %[[VAL_19:.*]] = llvm.insertvalue %[[VAL_5]], %[[VAL_18]][4] : !llvm.struct<"_Struct_always_inline_expr", packed (i1, i1, i1, i1, i1, i1, i1, i5, i5, array<2 x i5>, i1, array<2 x i5>)>
// CHECK: %[[VAL_20:.*]] = llvm.insertvalue %[[VAL_6]], %[[VAL_19]][5] : !llvm.struct<"_Struct_always_inline_expr", packed (i1, i1, i1, i1, i1, i1, i1, i5, i5, array<2 x i5>, i1, array<2 x i5>)>
// CHECK: %[[VAL_21:.*]] = llvm.insertvalue %[[VAL_7]], %[[VAL_20]][6] : !llvm.struct<"_Struct_always_inline_expr", packed (i1, i1, i1, i1, i1, i1, i1, i5, i5, array<2 x i5>, i1, array<2 x i5>)>
// CHECK: %[[VAL_22:.*]] = llvm.insertvalue %[[VAL_8]], %[[VAL_21]][7] : !llvm.struct<"_Struct_always_inline_expr", packed (i1, i1, i1, i1, i1, i1, i1, i5, i5, array<2 x i5>, i1, array<2 x i5>)>
// CHECK: %[[VAL_23:.*]] = llvm.insertvalue %[[VAL_9]], %[[VAL_22]][8] : !llvm.struct<"_Struct_always_inline_expr", packed (i1, i1, i1, i1, i1, i1, i1, i5, i5, array<2 x i5>, i1, array<2 x i5>)>
// CHECK: %[[VAL_24:.*]] = llvm.insertvalue %[[VAL_13]], %[[VAL_23]][9] : !llvm.struct<"_Struct_always_inline_expr", packed (i1, i1, i1, i1, i1, i1, i1, i5, i5, array<2 x i5>, i1, array<2 x i5>)>
// CHECK: %[[VAL_25:.*]] = llvm.insertvalue %[[VAL_4]], %[[VAL_24]][10] : !llvm.struct<"_Struct_always_inline_expr", packed (i1, i1, i1, i1, i1, i1, i1, i5, i5, array<2 x i5>, i1, array<2 x i5>)>
// CHECK: %[[VAL_26:.*]] = llvm.insertvalue %[[VAL_13]], %[[VAL_25]][11] : !llvm.struct<"_Struct_always_inline_expr", packed (i1, i1, i1, i1, i1, i1, i1, i5, i5, array<2 x i5>, i1, array<2 x i5>)>
// CHECK: llvm.return %[[VAL_26]] : !llvm.struct<"_Struct_always_inline_expr", packed (i1, i1, i1, i1, i1, i1, i1, i5, i5, array<2 x i5>, i1, array<2 x i5>)>
// CHECK: }

// CHECK-LABEL: llvm.mlir.global internal @_Struct_Moduleshift() {addr_space = 0 : i32} : !llvm.struct<"_Struct_Moduleshift", packed (i1, i1, i1, i1, ptr<struct<"_Struct_my_dff", packed (i1, i1, i1, i1, i1, i1)>>, ptr<struct<"_Struct_my_dff", packed (i1, i1, i1, i1, i1, i1)>>, ptr<struct<"_Struct_my_dff", packed (i1, i1, i1, i1, i1, i1)>>)> {
// CHECK: %[[VAL_0:.*]] = llvm.mlir.constant(false) : i1
// CHECK: %[[VAL_1:.*]] = llvm.mlir.constant(false) : i1
// CHECK: %[[VAL_2:.*]] = llvm.mlir.constant(false) : i1
// CHECK: %[[VAL_3:.*]] = llvm.mlir.constant(false) : i1
// CHECK: %[[VAL_4:.*]] = llvm.mlir.null : !llvm.ptr<struct<"_Struct_my_dff", packed (i1, i1, i1, i1, i1, i1)>>
// CHECK: %[[VAL_5:.*]] = llvm.mlir.null : !llvm.ptr<struct<"_Struct_my_dff", packed (i1, i1, i1, i1, i1, i1)>>
// CHECK: %[[VAL_6:.*]] = llvm.mlir.null : !llvm.ptr<struct<"_Struct_my_dff", packed (i1, i1, i1, i1, i1, i1)>>
// CHECK: %[[VAL_7:.*]] = llvm.mlir.undef : !llvm.struct<"_Struct_Moduleshift", packed (i1, i1, i1, i1, ptr<struct<"_Struct_my_dff", packed (i1, i1, i1, i1, i1, i1)>>, ptr<struct<"_Struct_my_dff", packed (i1, i1, i1, i1, i1, i1)>>, ptr<struct<"_Struct_my_dff", packed (i1, i1, i1, i1, i1, i1)>>)>
// CHECK: %[[VAL_8:.*]] = llvm.insertvalue %[[VAL_0]], %[[VAL_7]][0] : !llvm.struct<"_Struct_Moduleshift", packed (i1, i1, i1, i1, ptr<struct<"_Struct_my_dff", packed (i1, i1, i1, i1, i1, i1)>>, ptr<struct<"_Struct_my_dff", packed (i1, i1, i1, i1, i1, i1)>>, ptr<struct<"_Struct_my_dff", packed (i1, i1, i1, i1, i1, i1)>>)>
// CHECK: %[[VAL_9:.*]] = llvm.insertvalue %[[VAL_1]], %[[VAL_8]][1] : !llvm.struct<"_Struct_Moduleshift", packed (i1, i1, i1, i1, ptr<struct<"_Struct_my_dff", packed (i1, i1, i1, i1, i1, i1)>>, ptr<struct<"_Struct_my_dff", packed (i1, i1, i1, i1, i1, i1)>>, ptr<struct<"_Struct_my_dff", packed (i1, i1, i1, i1, i1, i1)>>)>
// CHECK: %[[VAL_10:.*]] = llvm.insertvalue %[[VAL_2]], %[[VAL_9]][2] : !llvm.struct<"_Struct_Moduleshift", packed (i1, i1, i1, i1, ptr<struct<"_Struct_my_dff", packed (i1, i1, i1, i1, i1, i1)>>, ptr<struct<"_Struct_my_dff", packed (i1, i1, i1, i1, i1, i1)>>, ptr<struct<"_Struct_my_dff", packed (i1, i1, i1, i1, i1, i1)>>)>
// CHECK: %[[VAL_11:.*]] = llvm.insertvalue %[[VAL_3]], %[[VAL_10]][3] : !llvm.struct<"_Struct_Moduleshift", packed (i1, i1, i1, i1, ptr<struct<"_Struct_my_dff", packed (i1, i1, i1, i1, i1, i1)>>, ptr<struct<"_Struct_my_dff", packed (i1, i1, i1, i1, i1, i1)>>, ptr<struct<"_Struct_my_dff", packed (i1, i1, i1, i1, i1, i1)>>)>
// CHECK: %[[VAL_12:.*]] = llvm.insertvalue %[[VAL_4]], %[[VAL_11]][4] : !llvm.struct<"_Struct_Moduleshift", packed (i1, i1, i1, i1, ptr<struct<"_Struct_my_dff", packed (i1, i1, i1, i1, i1, i1)>>, ptr<struct<"_Struct_my_dff", packed (i1, i1, i1, i1, i1, i1)>>, ptr<struct<"_Struct_my_dff", packed (i1, i1, i1, i1, i1, i1)>>)>
// CHECK: %[[VAL_13:.*]] = llvm.insertvalue %[[VAL_5]], %[[VAL_12]][5] : !llvm.struct<"_Struct_Moduleshift", packed (i1, i1, i1, i1, ptr<struct<"_Struct_my_dff", packed (i1, i1, i1, i1, i1, i1)>>, ptr<struct<"_Struct_my_dff", packed (i1, i1, i1, i1, i1, i1)>>, ptr<struct<"_Struct_my_dff", packed (i1, i1, i1, i1, i1, i1)>>)>
// CHECK: %[[VAL_14:.*]] = llvm.insertvalue %[[VAL_6]], %[[VAL_13]][6] : !llvm.struct<"_Struct_Moduleshift", packed (i1, i1, i1, i1, ptr<struct<"_Struct_my_dff", packed (i1, i1, i1, i1, i1, i1)>>, ptr<struct<"_Struct_my_dff", packed (i1, i1, i1, i1, i1, i1)>>, ptr<struct<"_Struct_my_dff", packed (i1, i1, i1, i1, i1, i1)>>)>
// CHECK: llvm.return %[[VAL_14]] : !llvm.struct<"_Struct_Moduleshift", packed (i1, i1, i1, i1, ptr<struct<"_Struct_my_dff", packed (i1, i1, i1, i1, i1, i1)>>, ptr<struct<"_Struct_my_dff", packed (i1, i1, i1, i1, i1, i1)>>, ptr<struct<"_Struct_my_dff", packed (i1, i1, i1, i1, i1, i1)>>)>
// CHECK: }

// CHECK-LABEL: llvm.mlir.global internal @_Struct_my_dff() {addr_space = 0 : i32} : !llvm.struct<"_Struct_my_dff", packed (i1, i1, i1, i1, i1, i1)> {
// CHECK: %[[VAL_0:.*]] = llvm.mlir.constant(false) : i1
// CHECK: %[[VAL_1:.*]] = llvm.mlir.constant(false) : i1
// CHECK: %[[VAL_2:.*]] = llvm.mlir.constant(false) : i1
// CHECK: %[[VAL_3:.*]] = llvm.mlir.constant(false) : i1
// CHECK: %[[VAL_4:.*]] = llvm.mlir.constant(false) : i1
// CHECK: %[[VAL_5:.*]] = llvm.mlir.constant(false) : i1
// CHECK: %[[VAL_6:.*]] = llvm.mlir.undef : !llvm.struct<"_Struct_my_dff", packed (i1, i1, i1, i1, i1, i1)>
// CHECK: %[[VAL_7:.*]] = llvm.insertvalue %[[VAL_0]], %[[VAL_6]][0] : !llvm.struct<"_Struct_my_dff", packed (i1, i1, i1, i1, i1, i1)>
// CHECK: %[[VAL_8:.*]] = llvm.insertvalue %[[VAL_2]], %[[VAL_7]][1] : !llvm.struct<"_Struct_my_dff", packed (i1, i1, i1, i1, i1, i1)>
// CHECK: %[[VAL_9:.*]] = llvm.insertvalue %[[VAL_3]], %[[VAL_8]][2] : !llvm.struct<"_Struct_my_dff", packed (i1, i1, i1, i1, i1, i1)>
// CHECK: %[[VAL_10:.*]] = llvm.insertvalue %[[VAL_4]], %[[VAL_9]][3] : !llvm.struct<"_Struct_my_dff", packed (i1, i1, i1, i1, i1, i1)>
// CHECK: %[[VAL_11:.*]] = llvm.insertvalue %[[VAL_1]], %[[VAL_10]][4] : !llvm.struct<"_Struct_my_dff", packed (i1, i1, i1, i1, i1, i1)>
// CHECK: %[[VAL_12:.*]] = llvm.insertvalue %[[VAL_5]], %[[VAL_11]][5] : !llvm.struct<"_Struct_my_dff", packed (i1, i1, i1, i1, i1, i1)>
// CHECK: llvm.return %[[VAL_12]] : !llvm.struct<"_Struct_my_dff", packed (i1, i1, i1, i1, i1, i1)>
// CHECK: }

// CHECK-LABEL: llvm.mlir.global internal @_Struct_test_case_3() {addr_space = 0 : i32} : !llvm.struct<"_Struct_test_case_3", packed (i1, i1, i1, i1, i4, i4, i4, i4, i4, i6, i4, i4, i4, i4, i6, i1, i4, i4, i4, i4, i6)> {
// CHECK: %[[VAL_0:.*]] = llvm.mlir.constant(false) : i1
// CHECK: %[[VAL_1:.*]] = llvm.mlir.constant(false) : i1
// CHECK: %[[VAL_2:.*]] = llvm.mlir.constant(false) : i1
// CHECK: %[[VAL_3:.*]] = llvm.mlir.constant(false) : i1
// CHECK: %[[VAL_4:.*]] = llvm.mlir.constant(false) : i1
// CHECK: %[[VAL_5:.*]] = llvm.mlir.constant(0 : i4) : i4
// CHECK: %[[VAL_6:.*]] = llvm.mlir.constant(0 : i4) : i4
// CHECK: %[[VAL_7:.*]] = llvm.mlir.constant(0 : i4) : i4
// CHECK: %[[VAL_8:.*]] = llvm.mlir.constant(0 : i4) : i4
// CHECK: %[[VAL_9:.*]] = llvm.mlir.constant(0 : i4) : i4
// CHECK: %[[VAL_10:.*]] = llvm.mlir.constant(0 : i6) : i6
// CHECK: %[[VAL_11:.*]] = llvm.mlir.constant(0 : i4) : i4
// CHECK: %[[VAL_12:.*]] = llvm.mlir.constant(0 : i4) : i4
// CHECK: %[[VAL_13:.*]] = llvm.mlir.constant(0 : i4) : i4
// CHECK: %[[VAL_14:.*]] = llvm.mlir.constant(0 : i4) : i4
// CHECK: %[[VAL_15:.*]] = llvm.mlir.constant(0 : i4) : i4
// CHECK: %[[VAL_16:.*]] = llvm.mlir.constant(0 : i4) : i4
// CHECK: %[[VAL_17:.*]] = llvm.mlir.constant(0 : i4) : i4
// CHECK: %[[VAL_18:.*]] = llvm.mlir.constant(0 : i4) : i4
// CHECK: %[[VAL_19:.*]] = llvm.mlir.constant(0 : i6) : i6
// CHECK: %[[VAL_20:.*]] = llvm.mlir.constant(0 : i6) : i6
// CHECK: %[[VAL_21:.*]] = llvm.mlir.undef : !llvm.struct<"_Struct_test_case_3", packed (i1, i1, i1, i1, i4, i4, i4, i4, i4, i6, i4, i4, i4, i4, i6, i1, i4, i4, i4, i4, i6)>
// CHECK: %[[VAL_22:.*]] = llvm.insertvalue %[[VAL_0]], %[[VAL_21]][0] : !llvm.struct<"_Struct_test_case_3", packed (i1, i1, i1, i1, i4, i4, i4, i4, i4, i6, i4, i4, i4, i4, i6, i1, i4, i4, i4, i4, i6)>
// CHECK: %[[VAL_23:.*]] = llvm.insertvalue %[[VAL_2]], %[[VAL_22]][1] : !llvm.struct<"_Struct_test_case_3", packed (i1, i1, i1, i1, i4, i4, i4, i4, i4, i6, i4, i4, i4, i4, i6, i1, i4, i4, i4, i4, i6)>
// CHECK: %[[VAL_24:.*]] = llvm.insertvalue %[[VAL_3]], %[[VAL_23]][2] : !llvm.struct<"_Struct_test_case_3", packed (i1, i1, i1, i1, i4, i4, i4, i4, i4, i6, i4, i4, i4, i4, i6, i1, i4, i4, i4, i4, i6)>
// CHECK: %[[VAL_25:.*]] = llvm.insertvalue %[[VAL_4]], %[[VAL_24]][3] : !llvm.struct<"_Struct_test_case_3", packed (i1, i1, i1, i1, i4, i4, i4, i4, i4, i6, i4, i4, i4, i4, i6, i1, i4, i4, i4, i4, i6)>
// CHECK: %[[VAL_26:.*]] = llvm.insertvalue %[[VAL_5]], %[[VAL_25]][4] : !llvm.struct<"_Struct_test_case_3", packed (i1, i1, i1, i1, i4, i4, i4, i4, i4, i6, i4, i4, i4, i4, i6, i1, i4, i4, i4, i4, i6)>
// CHECK: %[[VAL_27:.*]] = llvm.insertvalue %[[VAL_6]], %[[VAL_26]][5] : !llvm.struct<"_Struct_test_case_3", packed (i1, i1, i1, i1, i4, i4, i4, i4, i4, i6, i4, i4, i4, i4, i6, i1, i4, i4, i4, i4, i6)>
// CHECK: %[[VAL_28:.*]] = llvm.insertvalue %[[VAL_7]], %[[VAL_27]][6] : !llvm.struct<"_Struct_test_case_3", packed (i1, i1, i1, i1, i4, i4, i4, i4, i4, i6, i4, i4, i4, i4, i6, i1, i4, i4, i4, i4, i6)>
// CHECK: %[[VAL_29:.*]] = llvm.insertvalue %[[VAL_8]], %[[VAL_28]][7] : !llvm.struct<"_Struct_test_case_3", packed (i1, i1, i1, i1, i4, i4, i4, i4, i4, i6, i4, i4, i4, i4, i6, i1, i4, i4, i4, i4, i6)>
// CHECK: %[[VAL_30:.*]] = llvm.insertvalue %[[VAL_9]], %[[VAL_29]][8] : !llvm.struct<"_Struct_test_case_3", packed (i1, i1, i1, i1, i4, i4, i4, i4, i4, i6, i4, i4, i4, i4, i6, i1, i4, i4, i4, i4, i6)>
// CHECK: %[[VAL_31:.*]] = llvm.insertvalue %[[VAL_10]], %[[VAL_30]][9] : !llvm.struct<"_Struct_test_case_3", packed (i1, i1, i1, i1, i4, i4, i4, i4, i4, i6, i4, i4, i4, i4, i6, i1, i4, i4, i4, i4, i6)>
// CHECK: %[[VAL_32:.*]] = llvm.insertvalue %[[VAL_11]], %[[VAL_31]][10] : !llvm.struct<"_Struct_test_case_3", packed (i1, i1, i1, i1, i4, i4, i4, i4, i4, i6, i4, i4, i4, i4, i6, i1, i4, i4, i4, i4, i6)>
// CHECK: %[[VAL_33:.*]] = llvm.insertvalue %[[VAL_13]], %[[VAL_32]][11] : !llvm.struct<"_Struct_test_case_3", packed (i1, i1, i1, i1, i4, i4, i4, i4, i4, i6, i4, i4, i4, i4, i6, i1, i4, i4, i4, i4, i6)>
// CHECK: %[[VAL_34:.*]] = llvm.insertvalue %[[VAL_15]], %[[VAL_33]][12] : !llvm.struct<"_Struct_test_case_3", packed (i1, i1, i1, i1, i4, i4, i4, i4, i4, i6, i4, i4, i4, i4, i6, i1, i4, i4, i4, i4, i6)>
// CHECK: %[[VAL_35:.*]] = llvm.insertvalue %[[VAL_17]], %[[VAL_34]][13] : !llvm.struct<"_Struct_test_case_3", packed (i1, i1, i1, i1, i4, i4, i4, i4, i4, i6, i4, i4, i4, i4, i6, i1, i4, i4, i4, i4, i6)>
// CHECK: %[[VAL_36:.*]] = llvm.insertvalue %[[VAL_19]], %[[VAL_35]][14] : !llvm.struct<"_Struct_test_case_3", packed (i1, i1, i1, i1, i4, i4, i4, i4, i4, i6, i4, i4, i4, i4, i6, i1, i4, i4, i4, i4, i6)>
// CHECK: %[[VAL_37:.*]] = llvm.insertvalue %[[VAL_1]], %[[VAL_36]][15] : !llvm.struct<"_Struct_test_case_3", packed (i1, i1, i1, i1, i4, i4, i4, i4, i4, i6, i4, i4, i4, i4, i6, i1, i4, i4, i4, i4, i6)>
// CHECK: %[[VAL_38:.*]] = llvm.insertvalue %[[VAL_12]], %[[VAL_37]][16] : !llvm.struct<"_Struct_test_case_3", packed (i1, i1, i1, i1, i4, i4, i4, i4, i4, i6, i4, i4, i4, i4, i6, i1, i4, i4, i4, i4, i6)>
// CHECK: %[[VAL_39:.*]] = llvm.insertvalue %[[VAL_14]], %[[VAL_38]][17] : !llvm.struct<"_Struct_test_case_3", packed (i1, i1, i1, i1, i4, i4, i4, i4, i4, i6, i4, i4, i4, i4, i6, i1, i4, i4, i4, i4, i6)>
// CHECK: %[[VAL_40:.*]] = llvm.insertvalue %[[VAL_16]], %[[VAL_39]][18] : !llvm.struct<"_Struct_test_case_3", packed (i1, i1, i1, i1, i4, i4, i4, i4, i4, i6, i4, i4, i4, i4, i6, i1, i4, i4, i4, i4, i6)>
// CHECK: %[[VAL_41:.*]] = llvm.insertvalue %[[VAL_18]], %[[VAL_40]][19] : !llvm.struct<"_Struct_test_case_3", packed (i1, i1, i1, i1, i4, i4, i4, i4, i4, i6, i4, i4, i4, i4, i6, i1, i4, i4, i4, i4, i6)>
// CHECK: %[[VAL_42:.*]] = llvm.insertvalue %[[VAL_20]], %[[VAL_41]][20] : !llvm.struct<"_Struct_test_case_3", packed (i1, i1, i1, i1, i4, i4, i4, i4, i4, i6, i4, i4, i4, i4, i6, i1, i4, i4, i4, i4, i6)>
// CHECK: llvm.return %[[VAL_42]] : !llvm.struct<"_Struct_test_case_3", packed (i1, i1, i1, i1, i4, i4, i4, i4, i4, i6, i4, i4, i4, i4, i6, i1, i4, i4, i4, i4, i6)>
// CHECK: }

// CHECK-LABEL: llvm.mlir.global internal @_Struct_test_case_2() {addr_space = 0 : i32} : !llvm.struct<"_Struct_test_case_2", packed (i1, i1, i1, i1, i1, i1, i1, i1, i1, i1, i1, ptr<struct<"_Struct_SimJTAG", opaque>>, ptr<struct<"_Struct_ClockDividerN", opaque>>)> {
// CHECK: %[[VAL_0:.*]] = llvm.mlir.constant(false) : i1
// CHECK: %[[VAL_1:.*]] = llvm.mlir.constant(false) : i1
// CHECK: %[[VAL_2:.*]] = llvm.mlir.constant(false) : i1
// CHECK: %[[VAL_3:.*]] = llvm.mlir.constant(false) : i1
// CHECK: %[[VAL_4:.*]] = llvm.mlir.constant(false) : i1
// CHECK: %[[VAL_5:.*]] = llvm.mlir.constant(false) : i1
// CHECK: %[[VAL_6:.*]] = llvm.mlir.constant(false) : i1
// CHECK: %[[VAL_7:.*]] = llvm.mlir.constant(false) : i1
// CHECK: %[[VAL_8:.*]] = llvm.mlir.constant(false) : i1
// CHECK: %[[VAL_9:.*]] = llvm.mlir.constant(false) : i1
// CHECK: %[[VAL_10:.*]] = llvm.mlir.constant(false) : i1
// CHECK: %[[VAL_11:.*]] = llvm.mlir.null : !llvm.ptr<struct<"_Struct_SimJTAG", opaque>>
// CHECK: %[[VAL_12:.*]] = llvm.mlir.null : !llvm.ptr<struct<"_Struct_ClockDividerN", opaque>>
// CHECK: %[[VAL_13:.*]] = llvm.mlir.undef : !llvm.struct<"_Struct_test_case_2", packed (i1, i1, i1, i1, i1, i1, i1, i1, i1, i1, i1, ptr<struct<"_Struct_SimJTAG", opaque>>, ptr<struct<"_Struct_ClockDividerN", opaque>>)>
// CHECK: %[[VAL_14:.*]] = llvm.insertvalue %[[VAL_0]], %[[VAL_13]][0] : !llvm.struct<"_Struct_test_case_2", packed (i1, i1, i1, i1, i1, i1, i1, i1, i1, i1, i1, ptr<struct<"_Struct_SimJTAG", opaque>>, ptr<struct<"_Struct_ClockDividerN", opaque>>)>
// CHECK: %[[VAL_15:.*]] = llvm.insertvalue %[[VAL_2]], %[[VAL_14]][1] : !llvm.struct<"_Struct_test_case_2", packed (i1, i1, i1, i1, i1, i1, i1, i1, i1, i1, i1, ptr<struct<"_Struct_SimJTAG", opaque>>, ptr<struct<"_Struct_ClockDividerN", opaque>>)>
// CHECK: %[[VAL_16:.*]] = llvm.insertvalue %[[VAL_3]], %[[VAL_15]][2] : !llvm.struct<"_Struct_test_case_2", packed (i1, i1, i1, i1, i1, i1, i1, i1, i1, i1, i1, ptr<struct<"_Struct_SimJTAG", opaque>>, ptr<struct<"_Struct_ClockDividerN", opaque>>)>
// CHECK: %[[VAL_17:.*]] = llvm.insertvalue %[[VAL_4]], %[[VAL_16]][3] : !llvm.struct<"_Struct_test_case_2", packed (i1, i1, i1, i1, i1, i1, i1, i1, i1, i1, i1, ptr<struct<"_Struct_SimJTAG", opaque>>, ptr<struct<"_Struct_ClockDividerN", opaque>>)>
// CHECK: %[[VAL_18:.*]] = llvm.insertvalue %[[VAL_5]], %[[VAL_17]][4] : !llvm.struct<"_Struct_test_case_2", packed (i1, i1, i1, i1, i1, i1, i1, i1, i1, i1, i1, ptr<struct<"_Struct_SimJTAG", opaque>>, ptr<struct<"_Struct_ClockDividerN", opaque>>)>
// CHECK: %[[VAL_19:.*]] = llvm.insertvalue %[[VAL_6]], %[[VAL_18]][5] : !llvm.struct<"_Struct_test_case_2", packed (i1, i1, i1, i1, i1, i1, i1, i1, i1, i1, i1, ptr<struct<"_Struct_SimJTAG", opaque>>, ptr<struct<"_Struct_ClockDividerN", opaque>>)>
// CHECK: %[[VAL_20:.*]] = llvm.insertvalue %[[VAL_1]], %[[VAL_19]][6] : !llvm.struct<"_Struct_test_case_2", packed (i1, i1, i1, i1, i1, i1, i1, i1, i1, i1, i1, ptr<struct<"_Struct_SimJTAG", opaque>>, ptr<struct<"_Struct_ClockDividerN", opaque>>)>
// CHECK: %[[VAL_21:.*]] = llvm.insertvalue %[[VAL_7]], %[[VAL_20]][7] : !llvm.struct<"_Struct_test_case_2", packed (i1, i1, i1, i1, i1, i1, i1, i1, i1, i1, i1, ptr<struct<"_Struct_SimJTAG", opaque>>, ptr<struct<"_Struct_ClockDividerN", opaque>>)>
// CHECK: %[[VAL_22:.*]] = llvm.insertvalue %[[VAL_8]], %[[VAL_21]][8] : !llvm.struct<"_Struct_test_case_2", packed (i1, i1, i1, i1, i1, i1, i1, i1, i1, i1, i1, ptr<struct<"_Struct_SimJTAG", opaque>>, ptr<struct<"_Struct_ClockDividerN", opaque>>)>
// CHECK: %[[VAL_23:.*]] = llvm.insertvalue %[[VAL_9]], %[[VAL_22]][9] : !llvm.struct<"_Struct_test_case_2", packed (i1, i1, i1, i1, i1, i1, i1, i1, i1, i1, i1, ptr<struct<"_Struct_SimJTAG", opaque>>, ptr<struct<"_Struct_ClockDividerN", opaque>>)>
// CHECK: %[[VAL_24:.*]] = llvm.insertvalue %[[VAL_10]], %[[VAL_23]][10] : !llvm.struct<"_Struct_test_case_2", packed (i1, i1, i1, i1, i1, i1, i1, i1, i1, i1, i1, ptr<struct<"_Struct_SimJTAG", opaque>>, ptr<struct<"_Struct_ClockDividerN", opaque>>)>
// CHECK: %[[VAL_25:.*]] = llvm.insertvalue %[[VAL_11]], %[[VAL_24]][11] : !llvm.struct<"_Struct_test_case_2", packed (i1, i1, i1, i1, i1, i1, i1, i1, i1, i1, i1, ptr<struct<"_Struct_SimJTAG", opaque>>, ptr<struct<"_Struct_ClockDividerN", opaque>>)>
// CHECK: %[[VAL_26:.*]] = llvm.insertvalue %[[VAL_12]], %[[VAL_25]][12] : !llvm.struct<"_Struct_test_case_2", packed (i1, i1, i1, i1, i1, i1, i1, i1, i1, i1, i1, ptr<struct<"_Struct_SimJTAG", opaque>>, ptr<struct<"_Struct_ClockDividerN", opaque>>)>
// CHECK: llvm.return %[[VAL_26]] : !llvm.struct<"_Struct_test_case_2", packed (i1, i1, i1, i1, i1, i1, i1, i1, i1, i1, i1, ptr<struct<"_Struct_SimJTAG", opaque>>, ptr<struct<"_Struct_ClockDividerN", opaque>>)>
// CHECK: }

// CHECK-LABEL: llvm.mlir.global internal @_Struct_callee() {addr_space = 0 : i32} : !llvm.struct<"_Struct_callee", packed (i1, i1, i1, i1, i1, i1)> {
// CHECK: %[[VAL_0:.*]] = llvm.mlir.constant(false) : i1
// CHECK: %[[VAL_1:.*]] = llvm.mlir.constant(false) : i1
// CHECK: %[[VAL_2:.*]] = llvm.mlir.constant(false) : i1
// CHECK: %[[VAL_3:.*]] = llvm.mlir.constant(false) : i1
// CHECK: %[[VAL_4:.*]] = llvm.mlir.constant(false) : i1
// CHECK: %[[VAL_5:.*]] = llvm.mlir.constant(false) : i1
// CHECK: %[[VAL_6:.*]] = llvm.mlir.undef : !llvm.struct<"_Struct_callee", packed (i1, i1, i1, i1, i1, i1)>
// CHECK: %[[VAL_7:.*]] = llvm.insertvalue %[[VAL_0]], %[[VAL_6]][0] : !llvm.struct<"_Struct_callee", packed (i1, i1, i1, i1, i1, i1)>
// CHECK: %[[VAL_8:.*]] = llvm.insertvalue %[[VAL_2]], %[[VAL_7]][1] : !llvm.struct<"_Struct_callee", packed (i1, i1, i1, i1, i1, i1)>
// CHECK: %[[VAL_9:.*]] = llvm.insertvalue %[[VAL_3]], %[[VAL_8]][2] : !llvm.struct<"_Struct_callee", packed (i1, i1, i1, i1, i1, i1)>
// CHECK: %[[VAL_10:.*]] = llvm.insertvalue %[[VAL_4]], %[[VAL_9]][3] : !llvm.struct<"_Struct_callee", packed (i1, i1, i1, i1, i1, i1)>
// CHECK: %[[VAL_11:.*]] = llvm.insertvalue %[[VAL_1]], %[[VAL_10]][4] : !llvm.struct<"_Struct_callee", packed (i1, i1, i1, i1, i1, i1)>
// CHECK: %[[VAL_12:.*]] = llvm.insertvalue %[[VAL_5]], %[[VAL_11]][5] : !llvm.struct<"_Struct_callee", packed (i1, i1, i1, i1, i1, i1)>
// CHECK: llvm.return %[[VAL_12]] : !llvm.struct<"_Struct_callee", packed (i1, i1, i1, i1, i1, i1)>
// CHECK: }

// CHECK-LABEL: llvm.mlir.global internal @_Struct_test_case_1() {addr_space = 0 : i32} : !llvm.struct<"_Struct_test_case_1", packed (i1, i1, i1, i3, i1, i1, i1, ptr<struct<"_Struct_callee", packed (i1, i1, i1, i1, i1, i1)>>)> {
// CHECK: %[[VAL_0:.*]] = llvm.mlir.constant(false) : i1
// CHECK: %[[VAL_1:.*]] = llvm.mlir.constant(false) : i1
// CHECK: %[[VAL_2:.*]] = llvm.mlir.constant(false) : i1
// CHECK: %[[VAL_3:.*]] = llvm.mlir.constant(false) : i1
// CHECK: %[[VAL_4:.*]] = llvm.mlir.constant(0 : i3) : i3
// CHECK: %[[VAL_5:.*]] = llvm.mlir.constant(false) : i1
// CHECK: %[[VAL_6:.*]] = llvm.mlir.constant(false) : i1
// CHECK: %[[VAL_7:.*]] = llvm.mlir.null : !llvm.ptr<struct<"_Struct_callee", packed (i1, i1, i1, i1, i1, i1)>>
// CHECK: %[[VAL_8:.*]] = llvm.mlir.undef : !llvm.struct<"_Struct_test_case_1", packed (i1, i1, i1, i3, i1, i1, i1, ptr<struct<"_Struct_callee", packed (i1, i1, i1, i1, i1, i1)>>)>
// CHECK: %[[VAL_9:.*]] = llvm.insertvalue %[[VAL_0]], %[[VAL_8]][0] : !llvm.struct<"_Struct_test_case_1", packed (i1, i1, i1, i3, i1, i1, i1, ptr<struct<"_Struct_callee", packed (i1, i1, i1, i1, i1, i1)>>)>
// CHECK: %[[VAL_10:.*]] = llvm.insertvalue %[[VAL_1]], %[[VAL_9]][1] : !llvm.struct<"_Struct_test_case_1", packed (i1, i1, i1, i3, i1, i1, i1, ptr<struct<"_Struct_callee", packed (i1, i1, i1, i1, i1, i1)>>)>
// CHECK: %[[VAL_11:.*]] = llvm.insertvalue %[[VAL_2]], %[[VAL_10]][2] : !llvm.struct<"_Struct_test_case_1", packed (i1, i1, i1, i3, i1, i1, i1, ptr<struct<"_Struct_callee", packed (i1, i1, i1, i1, i1, i1)>>)>
// CHECK: %[[VAL_12:.*]] = llvm.insertvalue %[[VAL_4]], %[[VAL_11]][3] : !llvm.struct<"_Struct_test_case_1", packed (i1, i1, i1, i3, i1, i1, i1, ptr<struct<"_Struct_callee", packed (i1, i1, i1, i1, i1, i1)>>)>
// CHECK: %[[VAL_13:.*]] = llvm.insertvalue %[[VAL_3]], %[[VAL_12]][4] : !llvm.struct<"_Struct_test_case_1", packed (i1, i1, i1, i3, i1, i1, i1, ptr<struct<"_Struct_callee", packed (i1, i1, i1, i1, i1, i1)>>)>
// CHECK: %[[VAL_14:.*]] = llvm.insertvalue %[[VAL_5]], %[[VAL_13]][5] : !llvm.struct<"_Struct_test_case_1", packed (i1, i1, i1, i3, i1, i1, i1, ptr<struct<"_Struct_callee", packed (i1, i1, i1, i1, i1, i1)>>)>
// CHECK: %[[VAL_15:.*]] = llvm.insertvalue %[[VAL_6]], %[[VAL_14]][6] : !llvm.struct<"_Struct_test_case_1", packed (i1, i1, i1, i3, i1, i1, i1, ptr<struct<"_Struct_callee", packed (i1, i1, i1, i1, i1, i1)>>)>
// CHECK: %[[VAL_16:.*]] = llvm.insertvalue %[[VAL_7]], %[[VAL_15]][7] : !llvm.struct<"_Struct_test_case_1", packed (i1, i1, i1, i3, i1, i1, i1, ptr<struct<"_Struct_callee", packed (i1, i1, i1, i1, i1, i1)>>)>
// CHECK: llvm.return %[[VAL_16]] : !llvm.struct<"_Struct_test_case_1", packed (i1, i1, i1, i3, i1, i1, i1, ptr<struct<"_Struct_callee", packed (i1, i1, i1, i1, i1, i1)>>)>
// CHECK: }

// CHECK-LABEL: hw.module @test_case_1(
// CHECK-SAME: %[[VAL_0:.*]]: !llvm.ptr<struct<"_Struct_test_case_1", packed (i1, i1, i1, i3, i1, i1, i1, ptr<struct<"_Struct_callee", packed (i1, i1, i1, i1, i1, i1)>>)>>) attributes {sv.trigger = [2 : i8]} {
  hw.module @test_case_1(%arg0: i1, %arg1: i1, %arg2: i1, %arg3: i3) attributes {sv.trigger = [2 : i8]} {
    // CHECK: %[[VAL_1:.*]] = llvm.mlir.constant(0 : i32) : i32
    // CHECK: %[[VAL_2:.*]] = llvm.getelementptr inbounds %[[VAL_0]]{{\[}}%[[VAL_1]], 0] : (!llvm.ptr<struct<"_Struct_test_case_1", packed (i1, i1, i1, i3, i1, i1, i1, ptr<struct<"_Struct_callee", packed (i1, i1, i1, i1, i1, i1)>>)>>, i32) -> !llvm.ptr<i1>
    // CHECK: %[[VAL_3:.*]] = llvm.load %[[VAL_2]] : !llvm.ptr<i1>
    // CHECK: %[[VAL_4:.*]] = llvm.getelementptr inbounds %[[VAL_0]]{{\[}}%[[VAL_1]], 1] : (!llvm.ptr<struct<"_Struct_test_case_1", packed (i1, i1, i1, i3, i1, i1, i1, ptr<struct<"_Struct_callee", packed (i1, i1, i1, i1, i1, i1)>>)>>, i32) -> !llvm.ptr<i1>
    // CHECK: %[[VAL_5:.*]] = llvm.load %[[VAL_4]] : !llvm.ptr<i1>
    // CHECK: %[[VAL_6:.*]] = llvm.getelementptr inbounds %[[VAL_0]]{{\[}}%[[VAL_1]], 2] : (!llvm.ptr<struct<"_Struct_test_case_1", packed (i1, i1, i1, i3, i1, i1, i1, ptr<struct<"_Struct_callee", packed (i1, i1, i1, i1, i1, i1)>>)>>, i32) -> !llvm.ptr<i1>
    // CHECK: %[[VAL_7:.*]] = llvm.load %[[VAL_6]] : !llvm.ptr<i1>
    // CHECK: %[[VAL_8:.*]] = llvm.getelementptr inbounds %[[VAL_0]]{{\[}}%[[VAL_1]], 7] : (!llvm.ptr<struct<"_Struct_test_case_1", packed (i1, i1, i1, i3, i1, i1, i1, ptr<struct<"_Struct_callee", packed (i1, i1, i1, i1, i1, i1)>>)>>, i32) -> !llvm.ptr<ptr<struct<"_Struct_callee", packed (i1, i1, i1, i1, i1, i1)>>>
    // CHECK: %[[VAL_9:.*]] = llvm.load %[[VAL_8]] : !llvm.ptr<ptr<struct<"_Struct_callee", packed (i1, i1, i1, i1, i1, i1)>>>
    // CHECK: %[[VAL_10:.*]] = llvm.getelementptr inbounds %[[VAL_9]]{{\[}}%[[VAL_1]], 0] : (!llvm.ptr<struct<"_Struct_callee", packed (i1, i1, i1, i1, i1, i1)>>, i32) -> !llvm.ptr<i1>
    // CHECK: llvm.store %[[VAL_3]], %[[VAL_10]] : !llvm.ptr<i1>
    // CHECK: %[[VAL_11:.*]] = llvm.getelementptr inbounds %[[VAL_9]]{{\[}}%[[VAL_1]], 1] : (!llvm.ptr<struct<"_Struct_callee", packed (i1, i1, i1, i1, i1, i1)>>, i32) -> !llvm.ptr<i1>
    // CHECK: llvm.store %[[VAL_5]], %[[VAL_11]] : !llvm.ptr<i1>
    // CHECK: hw.instance "Calleex" @callee(ptr_struct_callee: %[[VAL_9]]: !llvm.ptr<struct<"_Struct_callee", packed (i1, i1, i1, i1, i1, i1)>>) -> ()
    %Calleex.out0, %Calleex.out1 = hw.instance "Calleex" @callee(arg0: %arg0: i1, arg1: %arg1: i1) -> (out0: i1, out1: i1) {sv.trigger = [0 : i8, 1 : i8]}

    // CHECK: %[[VAL_12:.*]] = llvm.mlir.constant(2 : i32) : i32
    // CHECK: %[[VAL_13:.*]] = llvm.getelementptr inbounds %[[VAL_9]]{{\[}}%[[VAL_1]], 2] : (!llvm.ptr<struct<"_Struct_callee", packed (i1, i1, i1, i1, i1, i1)>>, i32) -> !llvm.ptr<i1>
    // CHECK: %[[VAL_14:.*]] = llvm.load %[[VAL_13]] : !llvm.ptr<i1>
    // CHECK: %[[VAL_15:.*]] = llvm.mlir.constant(3 : i32) : i32
    // CHECK: %[[VAL_16:.*]] = llvm.getelementptr inbounds %[[VAL_9]]{{\[}}%[[VAL_1]], 3] : (!llvm.ptr<struct<"_Struct_callee", packed (i1, i1, i1, i1, i1, i1)>>, i32) -> !llvm.ptr<i1>
    // CHECK: %[[VAL_17:.*]] = llvm.load %[[VAL_16]] : !llvm.ptr<i1>
    // CHECK: sv.always posedge %[[VAL_14]], negedge %[[VAL_7]], edge %[[VAL_17]] {
    // CHECK: }
    sv.always posedge %Calleex.out0, negedge %arg2, edge %Calleex.out1 {
    }
    hw.output
  }

// CHECK-LABEL: hw.module @callee(
// CHECK-SAME: %[[VAL_0:.*]]: !llvm.ptr<struct<"_Struct_callee", packed (i1, i1, i1, i1, i1, i1)>>) attributes {sv.trigger = [0 : i8]} {
  hw.module @callee(%arg0: i1, %arg1: i1) -> (out0: i1, out1: i1) attributes {sv.trigger = [0 : i8]} {
    // CHECK: %[[VAL_1:.*]] = llvm.mlir.constant(0 : i32) : i32
    // CHECK: %[[VAL_2:.*]] = llvm.getelementptr inbounds %[[VAL_0]]{{\[}}%[[VAL_1]], 0] : (!llvm.ptr<struct<"_Struct_callee", packed (i1, i1, i1, i1, i1, i1)>>, i32) -> !llvm.ptr<i1>
    // CHECK: %[[VAL_3:.*]] = llvm.load %[[VAL_2]] : !llvm.ptr<i1>
    // CHECK: %[[VAL_4:.*]] = llvm.getelementptr inbounds %[[VAL_0]]{{\[}}%[[VAL_1]], 1] : (!llvm.ptr<struct<"_Struct_callee", packed (i1, i1, i1, i1, i1, i1)>>, i32) -> !llvm.ptr<i1>
    // CHECK: %[[VAL_5:.*]] = llvm.load %[[VAL_4]] : !llvm.ptr<i1>

    // CHECK: %[[VAL_6:.*]] = llvm.mlir.constant(true) : i1
    %0 = llvm.mlir.constant(true) : i1

    // CHECK: %[[VAL_7:.*]] = comb.or %[[VAL_6]], %[[VAL_3]] : i1
    // CHECK: %[[VAL_8:.*]] = comb.and %[[VAL_7]], %[[VAL_5]] {sv.trigger = [0 : i8]} : i1
    %1 = comb.or %0, %arg0 : i1
    %2 = comb.and %1, %arg1 {sv.trigger = [0 : i8]} : i1

    // CHECK: sv.always posedge %[[VAL_3]], negedge %[[VAL_8]] {
    // CHECK: }
    sv.always posedge %arg0, negedge %2 {
    }

    // CHECK: %[[VAL_9:.*]] = llvm.getelementptr inbounds %[[VAL_0]]{{\[}}%[[VAL_1]], 2] : (!llvm.ptr<struct<"_Struct_callee", packed (i1, i1, i1, i1, i1, i1)>>, i32) -> !llvm.ptr<i1>
    // CHECK: llvm.store %[[VAL_7]], %[[VAL_9]] : !llvm.ptr<i1>
    // CHECK: %[[VAL_10:.*]] = llvm.getelementptr inbounds %[[VAL_0]]{{\[}}%[[VAL_1]], 3] : (!llvm.ptr<struct<"_Struct_callee", packed (i1, i1, i1, i1, i1, i1)>>, i32) -> !llvm.ptr<i1>
    // CHECK: llvm.store %[[VAL_8]], %[[VAL_10]] : !llvm.ptr<i1>
    // CHECK: hw.output
    // CHECK: }
    hw.output %1, %2 : i1, i1
  }

// CHECK-LABEL: hw.module @test_case_2(
// CHECK-SAME: %[[VAL_0:.*]]: !llvm.ptr<struct<"_Struct_test_case_2", packed (i1, i1, i1, i1, i1, i1, i1, i1, i1, i1, i1, ptr<struct<"_Struct_SimJTAG", opaque>>, ptr<struct<"_Struct_ClockDividerN", opaque>>)>>) attributes {sv.trigger = [0 : i8]} {
  hw.module @test_case_2(%clock: i1, %reset: i1, %TDO: i1, %enable: i1) -> (out0: i1, out1: i1) attributes {sv.trigger = [0 : i8]} {
    // CHECK: %[[VAL_1:.*]] = llvm.mlir.constant(0 : i32) : i32
    // CHECK: %[[VAL_2:.*]] = llvm.getelementptr inbounds %[[VAL_0]]{{\[}}%[[VAL_1]], 0] : (!llvm.ptr<struct<"_Struct_test_case_2", packed (i1, i1, i1, i1, i1, i1, i1, i1, i1, i1, i1, ptr<struct<"_Struct_SimJTAG", opaque>>, ptr<struct<"_Struct_ClockDividerN", opaque>>)>>, i32) -> !llvm.ptr<i1>
    // CHECK: %[[VAL_3:.*]] = llvm.load %[[VAL_2]] : !llvm.ptr<i1>
    // CHECK: %[[VAL_4:.*]] = llvm.getelementptr inbounds %[[VAL_0]]{{\[}}%[[VAL_1]], 1] : (!llvm.ptr<struct<"_Struct_test_case_2", packed (i1, i1, i1, i1, i1, i1, i1, i1, i1, i1, i1, ptr<struct<"_Struct_SimJTAG", opaque>>, ptr<struct<"_Struct_ClockDividerN", opaque>>)>>, i32) -> !llvm.ptr<i1>
    // CHECK: %[[VAL_5:.*]] = llvm.load %[[VAL_4]] : !llvm.ptr<i1>
    // CHECK: %[[VAL_6:.*]] = llvm.getelementptr inbounds %[[VAL_0]]{{\[}}%[[VAL_1]], 2] : (!llvm.ptr<struct<"_Struct_test_case_2", packed (i1, i1, i1, i1, i1, i1, i1, i1, i1, i1, i1, ptr<struct<"_Struct_SimJTAG", opaque>>, ptr<struct<"_Struct_ClockDividerN", opaque>>)>>, i32) -> !llvm.ptr<i1>
    // CHECK: %[[VAL_7:.*]] = llvm.load %[[VAL_6]] : !llvm.ptr<i1>
    // CHECK: %[[VAL_8:.*]] = llvm.getelementptr inbounds %[[VAL_0]]{{\[}}%[[VAL_1]], 3] : (!llvm.ptr<struct<"_Struct_test_case_2", packed (i1, i1, i1, i1, i1, i1, i1, i1, i1, i1, i1, ptr<struct<"_Struct_SimJTAG", opaque>>, ptr<struct<"_Struct_ClockDividerN", opaque>>)>>, i32) -> !llvm.ptr<i1>
    // CHECK: %[[VAL_9:.*]] = llvm.load %[[VAL_8]] : !llvm.ptr<i1>

    // CHECK: %[[VAL_10:.*]] = llvm.mlir.constant(false) : i1
    // CHECK: %[[VAL_11:.*]] = llvm.mlir.constant(true) : i1
    %0 = llvm.mlir.constant(false) : i1
    %1 = llvm.mlir.constant(true) : i1
    // CHECK: %[[VAL_12:.*]] = comb.and %[[VAL_10]], %[[VAL_11]] : i1
    %2 = comb.and %0, %1 : i1

    // CHECK: %[[VAL_13:.*]] = llvm.getelementptr inbounds %[[VAL_0]]{{\[}}%[[VAL_1]], 11] : (!llvm.ptr<struct<"_Struct_test_case_2", packed (i1, i1, i1, i1, i1, i1, i1, i1, i1, i1, i1, ptr<struct<"_Struct_SimJTAG", opaque>>, ptr<struct<"_Struct_ClockDividerN", opaque>>)>>, i32) -> !llvm.ptr<ptr<struct<"_Struct_SimJTAG", opaque>>>
    // CHECK: %[[VAL_14:.*]] = llvm.load %[[VAL_13]] : !llvm.ptr<ptr<struct<"_Struct_SimJTAG", opaque>>>
    // CHECK: %[[VAL_15:.*]] = llvm.bitcast %[[VAL_14]] : !llvm.ptr<struct<"_Struct_SimJTAG", opaque>> to !llvm.ptr<struct<packed (i1, i1, i1, i1, i1, i1, i1, i1, i1, i1, i32)>>
    // CHECK: %[[VAL_16:.*]] = llvm.getelementptr inbounds %[[VAL_15]]{{\[}}%[[VAL_1]], 0] : (!llvm.ptr<struct<packed (i1, i1, i1, i1, i1, i1, i1, i1, i1, i1, i32)>>, i32) -> !llvm.ptr<i1>
    // CHECK: llvm.store %[[VAL_3]], %[[VAL_16]] : !llvm.ptr<i1>
    // CHECK: %[[VAL_17:.*]] = llvm.getelementptr inbounds %[[VAL_15]]{{\[}}%[[VAL_1]], 1] : (!llvm.ptr<struct<packed (i1, i1, i1, i1, i1, i1, i1, i1, i1, i1, i32)>>, i32) -> !llvm.ptr<i1>
    // CHECK: llvm.store %[[VAL_5]], %[[VAL_17]] : !llvm.ptr<i1>
    // CHECK: %[[VAL_18:.*]] = llvm.getelementptr inbounds %[[VAL_15]]{{\[}}%[[VAL_1]], 2] : (!llvm.ptr<struct<packed (i1, i1, i1, i1, i1, i1, i1, i1, i1, i1, i32)>>, i32) -> !llvm.ptr<i1>
    // CHECK: llvm.store %[[VAL_7]], %[[VAL_18]] : !llvm.ptr<i1>
    // CHECK: %[[VAL_19:.*]] = llvm.getelementptr inbounds %[[VAL_15]]{{\[}}%[[VAL_1]], 3] : (!llvm.ptr<struct<packed (i1, i1, i1, i1, i1, i1, i1, i1, i1, i1, i32)>>, i32) -> !llvm.ptr<i1>
    // CHECK: llvm.store %[[VAL_11]], %[[VAL_19]] : !llvm.ptr<i1>
    // CHECK: %[[VAL_20:.*]] = llvm.getelementptr inbounds %[[VAL_15]]{{\[}}%[[VAL_1]], 4] : (!llvm.ptr<struct<packed (i1, i1, i1, i1, i1, i1, i1, i1, i1, i1, i32)>>, i32) -> !llvm.ptr<i1>
    // CHECK: llvm.store %[[VAL_9]], %[[VAL_20]] : !llvm.ptr<i1>
    // CHECK: %[[VAL_21:.*]] = llvm.getelementptr inbounds %[[VAL_15]]{{\[}}%[[VAL_1]], 5] : (!llvm.ptr<struct<packed (i1, i1, i1, i1, i1, i1, i1, i1, i1, i1, i32)>>, i32) -> !llvm.ptr<i1>
    // CHECK: llvm.store %[[VAL_10]], %[[VAL_21]] : !llvm.ptr<i1>
    // CHECK: hw.instance "SimJTAG" @SimJTAG(ptr_struct_SimJTAG: %[[VAL_14]]: !llvm.ptr<struct<"_Struct_SimJTAG", opaque>>) -> ()
    %SimJTAG.jtag_TRSTn, %SimJTAG.jtag_TCK, %SimJTAG.jtag_TMS, %SimJTAG.jtag_TDI, %SimJTAG.exit = hw.instance "SimJTAG" @SimJTAG(clock: %clock: i1, reset: %reset: i1, jtag_TDO_data: %TDO: i1, jtag_TDO_driven: %1: i1, enable: %enable: i1,
                                                                                                                                init_done: %0: i1) -> (jtag_TRSTn: i1, jtag_TCK: i1, jtag_TMS: i1, jtag_TDI: i1, exit: i32) {sv.trigger = [1 : i8, 2 : i8, 3 : i8]}
    // CHECK: %[[VAL_22:.*]] = llvm.bitcast %[[VAL_14]] : !llvm.ptr<struct<"_Struct_SimJTAG", opaque>> to !llvm.ptr<struct<packed (i1, i1, i1, i1, i1, i1, i1, i1, i1, i1, i32)>>
    // CHECK: %[[VAL_23:.*]] = llvm.mlir.constant(6 : i32) : i32
    // CHECK: %[[VAL_24:.*]] = llvm.getelementptr inbounds %[[VAL_22]]{{\[}}%[[VAL_1]], 6] : (!llvm.ptr<struct<packed (i1, i1, i1, i1, i1, i1, i1, i1, i1, i1, i32)>>, i32) -> !llvm.ptr<i1>
    // CHECK: %[[VAL_25:.*]] = llvm.load %[[VAL_24]] : !llvm.ptr<i1>
    // CHECK: %[[VAL_26:.*]] = llvm.mlir.constant(7 : i32) : i32
    // CHECK: %[[VAL_27:.*]] = llvm.getelementptr inbounds %[[VAL_22]]{{\[}}%[[VAL_1]], 7] : (!llvm.ptr<struct<packed (i1, i1, i1, i1, i1, i1, i1, i1, i1, i1, i32)>>, i32) -> !llvm.ptr<i1>
    // CHECK: %[[VAL_28:.*]] = llvm.load %[[VAL_27]] : !llvm.ptr<i1>
    // CHECK: %[[VAL_29:.*]] = llvm.mlir.constant(8 : i32) : i32
    // CHECK: %[[VAL_30:.*]] = llvm.getelementptr inbounds %[[VAL_22]]{{\[}}%[[VAL_1]], 8] : (!llvm.ptr<struct<packed (i1, i1, i1, i1, i1, i1, i1, i1, i1, i1, i32)>>, i32) -> !llvm.ptr<i1>
    // CHECK: %[[VAL_31:.*]] = llvm.load %[[VAL_30]] : !llvm.ptr<i1>
    // CHECK: %[[VAL_32:.*]] = llvm.mlir.constant(9 : i32) : i32
    // CHECK: %[[VAL_33:.*]] = llvm.getelementptr inbounds %[[VAL_22]]{{\[}}%[[VAL_1]], 9] : (!llvm.ptr<struct<packed (i1, i1, i1, i1, i1, i1, i1, i1, i1, i1, i32)>>, i32) -> !llvm.ptr<i1>
    // CHECK: %[[VAL_34:.*]] = llvm.load %[[VAL_33]] : !llvm.ptr<i1>
    // CHECK: %[[VAL_35:.*]] = llvm.mlir.constant(10 : i32) : i32
    // CHECK: %[[VAL_36:.*]] = llvm.getelementptr inbounds %[[VAL_22]]{{\[}}%[[VAL_1]], 10] : (!llvm.ptr<struct<packed (i1, i1, i1, i1, i1, i1, i1, i1, i1, i1, i32)>>, i32) -> !llvm.ptr<i32>
    // CHECK: %[[VAL_37:.*]] = llvm.load %[[VAL_36]] : !llvm.ptr<i32>

    // CHECK: %[[VAL_38:.*]] = llvm.getelementptr inbounds %[[VAL_0]]{{\[}}%[[VAL_1]], 12] : (!llvm.ptr<struct<"_Struct_test_case_2", packed (i1, i1, i1, i1, i1, i1, i1, i1, i1, i1, i1, ptr<struct<"_Struct_SimJTAG", opaque>>, ptr<struct<"_Struct_ClockDividerN", opaque>>)>>, i32) -> !llvm.ptr<ptr<struct<"_Struct_ClockDividerN", opaque>>>
    // CHECK: %[[VAL_39:.*]] = llvm.load %[[VAL_38]] : !llvm.ptr<ptr<struct<"_Struct_ClockDividerN", opaque>>>
    // CHECK: %[[VAL_40:.*]] = llvm.bitcast %[[VAL_39]] : !llvm.ptr<struct<"_Struct_ClockDividerN", opaque>> to !llvm.ptr<struct<packed (i1, i1)>>
    // CHECK: %[[VAL_41:.*]] = llvm.getelementptr inbounds %[[VAL_40]]{{\[}}%[[VAL_1]], 0] : (!llvm.ptr<struct<packed (i1, i1)>>, i32) -> !llvm.ptr<i1>
    // CHECK: llvm.store %[[VAL_3]], %[[VAL_41]] : !llvm.ptr<i1>
    // CHECK: hw.instance "clockDividerN" @ClockDividerN(ptr_struct_ClockDividerN: %[[VAL_39]]: !llvm.ptr<struct<"_Struct_ClockDividerN", opaque>>) -> ()
    %clockDividerN.clk_out = hw.instance "clockDividerN" @ClockDividerN(clk_in: %clock: i1) -> (clk_out: i1) {sv.trigger = [0 : i8]}

    // CHECK: %[[VAL_42:.*]] = llvm.bitcast %[[VAL_39]] : !llvm.ptr<struct<"_Struct_ClockDividerN", opaque>> to !llvm.ptr<struct<packed (i1, i1)>>
    // CHECK: %[[VAL_43:.*]] = llvm.mlir.constant(1 : i32) : i32
    // CHECK: %[[VAL_44:.*]] = llvm.getelementptr inbounds %[[VAL_42]]{{\[}}%[[VAL_1]], 1] : (!llvm.ptr<struct<packed (i1, i1)>>, i32) -> !llvm.ptr<i1>
    // CHECK: %[[VAL_45:.*]] = llvm.load %[[VAL_44]] : !llvm.ptr<i1>
    // CHECK: sv.always posedge %[[VAL_3]], posedge %[[VAL_28]], posedge %[[VAL_45]] {
    // CHECK:   sv.if %[[VAL_5]] {
    // CHECK:   }
    // CHECK: }
    sv.always posedge %clock, posedge %SimJTAG.jtag_TCK, posedge %clockDividerN.clk_out {
      sv.if %reset {
      }
    }

    // CHECK: sv.always posedge %[[VAL_28]], negedge %[[VAL_34]], edge %[[VAL_31]] {
    // CHECK:   sv.if %[[VAL_25]] {
    // CHECK:   }
    // CHECK: }
    sv.always posedge %SimJTAG.jtag_TCK, negedge %SimJTAG.jtag_TDI, edge %SimJTAG.jtag_TMS {
      sv.if %SimJTAG.jtag_TRSTn {
      }
    }

    // CHECK: %[[VAL_46:.*]] = llvm.getelementptr inbounds %[[VAL_0]]{{\[}}%[[VAL_1]], 4] : (!llvm.ptr<struct<"_Struct_test_case_2", packed (i1, i1, i1, i1, i1, i1, i1, i1, i1, i1, i1, ptr<struct<"_Struct_SimJTAG", opaque>>, ptr<struct<"_Struct_ClockDividerN", opaque>>)>>, i32) -> !llvm.ptr<i1>
    // CHECK: llvm.store %[[VAL_31]], %[[VAL_46]] : !llvm.ptr<i1>
    // CHECK: %[[VAL_47:.*]] = llvm.getelementptr inbounds %[[VAL_0]]{{\[}}%[[VAL_1]], 5] : (!llvm.ptr<struct<"_Struct_test_case_2", packed (i1, i1, i1, i1, i1, i1, i1, i1, i1, i1, i1, ptr<struct<"_Struct_SimJTAG", opaque>>, ptr<struct<"_Struct_ClockDividerN", opaque>>)>>, i32) -> !llvm.ptr<i1>
    // CHECK: llvm.store %[[VAL_34]], %[[VAL_47]] : !llvm.ptr<i1>
    // CHECK: hw.output
    // CHECK: }
    hw.output %SimJTAG.jtag_TMS, %SimJTAG.jtag_TDI : i1, i1
  }

    // CHECK: hw.module.extern private @SimJTAG(%[[VAL_48:.*]]: !llvm.ptr<struct<"_Struct_SimJTAG", opaque>>)
    // CHECK: hw.module.extern private @ClockDividerN(%[[VAL_49:.*]]: !llvm.ptr<struct<"_Struct_ClockDividerN", opaque>>)
  hw.module.extern private @SimJTAG(%clock: i1, %reset: i1, %jtag_TDO_data: i1, %jtag_TDO_driven: i1, %enable: i1, %init_done: i1) -> (jtag_TRSTn: i1, jtag_TCK: i1, jtag_TMS: i1, jtag_TDI: i1, exit: i32)
  hw.module.extern private @ClockDividerN(%clk_in: i1) -> (clk_out: i1)

// CHECK-LABEL: hw.module @test_case_3(
// CHECK-SAME: %[[VAL_0:.*]]: !llvm.ptr<struct<"_Struct_test_case_3", packed (i1, i1, i1, i1, i4, i4, i4, i4, i4, i6, i4, i4, i4, i4, i6, i1, i4, i4, i4, i4, i6)>>) attributes {sv.trigger = [0 : i8]} {
  hw.module @test_case_3(%clock: i1, %reset: i1, %enable: i1, %ureset: i1, %d: i4) -> (Q_0: i4, Q_1: i4, Q_2: i4, Q_3: i4, Q_4: i6) attributes {sv.trigger = [0 : i8]} {
    // CHECK: %[[VAL_1:.*]] = llvm.mlir.constant(0 : i32) : i32
    // CHECK: %[[VAL_2:.*]] = llvm.getelementptr inbounds %[[VAL_0]]{{\[}}%[[VAL_1]], 0] : (!llvm.ptr<struct<"_Struct_test_case_3", packed (i1, i1, i1, i1, i4, i4, i4, i4, i4, i6, i4, i4, i4, i4, i6, i1, i4, i4, i4, i4, i6)>>, i32) -> !llvm.ptr<i1>
    // CHECK: %[[VAL_3:.*]] = llvm.load %[[VAL_2]] : !llvm.ptr<i1>
    // CHECK: %[[VAL_4:.*]] = llvm.getelementptr inbounds %[[VAL_0]]{{\[}}%[[VAL_1]], 3] : (!llvm.ptr<struct<"_Struct_test_case_3", packed (i1, i1, i1, i1, i4, i4, i4, i4, i4, i6, i4, i4, i4, i4, i6, i1, i4, i4, i4, i4, i6)>>, i32) -> !llvm.ptr<i1>
    // CHECK: %[[VAL_5:.*]] = llvm.load %[[VAL_4]] : !llvm.ptr<i1>
    // CHECK: %[[VAL_6:.*]] = llvm.getelementptr inbounds %[[VAL_0]]{{\[}}%[[VAL_1]], 4] : (!llvm.ptr<struct<"_Struct_test_case_3", packed (i1, i1, i1, i1, i4, i4, i4, i4, i4, i6, i4, i4, i4, i4, i6, i1, i4, i4, i4, i4, i6)>>, i32) -> !llvm.ptr<i4>
    // CHECK: %[[VAL_7:.*]] = llvm.load %[[VAL_6]] : !llvm.ptr<i4>

    // CHECK: %[[VAL_8:.*]] = hw.constant 0 : i4
    %c0_i4 = hw.constant 0 : i4

    // CHECK: %[[VAL_9:.*]] = sv.reg  : !hw.inout<i4>
    // CHECK: %[[VAL_10:.*]] = sv.read_inout %[[VAL_9]] : !hw.inout<i4>
    // CHECK: %[[VAL_11:.*]] = sv.reg  : !hw.inout<i4>
    // CHECK: %[[VAL_12:.*]] = sv.read_inout %[[VAL_11]] : !hw.inout<i4>
    // CHECK: %[[VAL_13:.*]] = sv.reg  : !hw.inout<i4>
    // CHECK: %[[VAL_14:.*]] = sv.read_inout %[[VAL_13]] : !hw.inout<i4>
    // CHECK: %[[VAL_15:.*]] = sv.reg  : !hw.inout<i4>
    // CHECK: %[[VAL_16:.*]] = sv.read_inout %[[VAL_15]] : !hw.inout<i4>
    // CHECK: %[[VAL_17:.*]] = sv.reg  : !hw.inout<i6>
    // CHECK: %[[VAL_18:.*]] = sv.read_inout %[[VAL_17]] : !hw.inout<i6>
    %Q_out_reg_0 = sv.reg  : !hw.inout<i4>
    %0 = sv.read_inout %Q_out_reg_0 : !hw.inout<i4>
    %Q_out_reg_1 = sv.reg  : !hw.inout<i4>
    %1 = sv.read_inout %Q_out_reg_1 : !hw.inout<i4>
    %Q_out_reg_2 = sv.reg  : !hw.inout<i4>
    %2 = sv.read_inout %Q_out_reg_2 : !hw.inout<i4>
    %Q_out_reg_3 = sv.reg  : !hw.inout<i4>
    %3 = sv.read_inout %Q_out_reg_3 : !hw.inout<i4>
    %test_reg_4 = sv.reg  : !hw.inout<i6>
    %4 = sv.read_inout %test_reg_4 : !hw.inout<i6>

    // CHECK: sv.always posedge %[[VAL_3]] {
    // CHECK:   sv.if %[[VAL_5]] {
    // CHECK:     sv.passign %[[VAL_9]], %[[VAL_8]] : i4
    // CHECK:     sv.passign %[[VAL_11]], %[[VAL_8]] : i4
    // CHECK:     sv.passign %[[VAL_13]], %[[VAL_8]] : i4
    // CHECK:     sv.passign %[[VAL_15]], %[[VAL_8]] : i4
    // CHECK:   } else {
    // CHECK:     sv.passign %[[VAL_9]], %[[VAL_7]] : i4
    // CHECK:     sv.passign %[[VAL_11]], %[[VAL_7]] : i4
    // CHECK:     sv.passign %[[VAL_13]], %[[VAL_7]] : i4
    // CHECK:     sv.passign %[[VAL_15]], %[[VAL_7]] : i4
    // CHECK:   }
    // CHECK: }
    sv.always posedge %clock {
      sv.if %ureset {
        sv.passign %Q_out_reg_0, %c0_i4 : i4
        sv.passign %Q_out_reg_1, %c0_i4 : i4
        sv.passign %Q_out_reg_2, %c0_i4 : i4
        sv.passign %Q_out_reg_3, %c0_i4 : i4
      } else {
        sv.passign %Q_out_reg_0, %d : i4
        sv.passign %Q_out_reg_1, %d : i4
        sv.passign %Q_out_reg_2, %d : i4
        sv.passign %Q_out_reg_3, %d : i4
      }
    }
    // CHECK: %[[VAL_19:.*]] = llvm.getelementptr inbounds %[[VAL_0]]{{\[}}%[[VAL_1]], 5] : (!llvm.ptr<struct<"_Struct_test_case_3", packed (i1, i1, i1, i1, i4, i4, i4, i4, i4, i6, i4, i4, i4, i4, i6, i1, i4, i4, i4, i4, i6)>>, i32) -> !llvm.ptr<i4>
    // CHECK: llvm.store %[[VAL_10]], %[[VAL_19]] : !llvm.ptr<i4>
    // CHECK: %[[VAL_20:.*]] = llvm.getelementptr inbounds %[[VAL_0]]{{\[}}%[[VAL_1]], 6] : (!llvm.ptr<struct<"_Struct_test_case_3", packed (i1, i1, i1, i1, i4, i4, i4, i4, i4, i6, i4, i4, i4, i4, i6, i1, i4, i4, i4, i4, i6)>>, i32) -> !llvm.ptr<i4>
    // CHECK: llvm.store %[[VAL_12]], %[[VAL_20]] : !llvm.ptr<i4>
    // CHECK: %[[VAL_21:.*]] = llvm.getelementptr inbounds %[[VAL_0]]{{\[}}%[[VAL_1]], 7] : (!llvm.ptr<struct<"_Struct_test_case_3", packed (i1, i1, i1, i1, i4, i4, i4, i4, i4, i6, i4, i4, i4, i4, i6, i1, i4, i4, i4, i4, i6)>>, i32) -> !llvm.ptr<i4>
    // CHECK: llvm.store %[[VAL_14]], %[[VAL_21]] : !llvm.ptr<i4>
    // CHECK: %[[VAL_22:.*]] = llvm.getelementptr inbounds %[[VAL_0]]{{\[}}%[[VAL_1]], 8] : (!llvm.ptr<struct<"_Struct_test_case_3", packed (i1, i1, i1, i1, i4, i4, i4, i4, i4, i6, i4, i4, i4, i4, i6, i1, i4, i4, i4, i4, i6)>>, i32) -> !llvm.ptr<i4>
    // CHECK: llvm.store %[[VAL_16]], %[[VAL_22]] : !llvm.ptr<i4>
    // CHECK: %[[VAL_23:.*]] = llvm.getelementptr inbounds %[[VAL_0]]{{\[}}%[[VAL_1]], 9] : (!llvm.ptr<struct<"_Struct_test_case_3", packed (i1, i1, i1, i1, i4, i4, i4, i4, i4, i6, i4, i4, i4, i4, i6, i1, i4, i4, i4, i4, i6)>>, i32) -> !llvm.ptr<i6>
    // CHECK: llvm.store %[[VAL_18]], %[[VAL_23]] : !llvm.ptr<i6>
    // CHECK: hw.output
    // CHECK: }
    hw.output %0, %1, %2, %3, %4 : i4, i4, i4, i4, i6
  }

// CHECK-LABEL: hw.module private @my_dff(
// CHECK-SAME: %[[VAL_0:.*]]: !llvm.ptr<struct<"_Struct_my_dff", packed (i1, i1, i1, i1, i1, i1)>>) attributes {sv.trigger = [0 : i8]} {
  hw.module private @my_dff(%clock: i1, %d: i1) -> (q: i1) attributes {sv.trigger = [0 : i8]} {
    // CHECK: %[[VAL_1:.*]] = llvm.mlir.constant(0 : i32) : i32
    // CHECK: %[[VAL_2:.*]] = llvm.getelementptr inbounds %[[VAL_0]]{{\[}}%[[VAL_1]], 0] : (!llvm.ptr<struct<"_Struct_my_dff", packed (i1, i1, i1, i1, i1, i1)>>, i32) -> !llvm.ptr<i1>
    // CHECK: %[[VAL_3:.*]] = llvm.load %[[VAL_2]] : !llvm.ptr<i1>
    // CHECK: %[[VAL_4:.*]] = llvm.getelementptr inbounds %[[VAL_0]]{{\[}}%[[VAL_1]], 1] : (!llvm.ptr<struct<"_Struct_my_dff", packed (i1, i1, i1, i1, i1, i1)>>, i32) -> !llvm.ptr<i1>
    // CHECK: %[[VAL_5:.*]] = llvm.load %[[VAL_4]] : !llvm.ptr<i1>

    // CHECK: %[[VAL_6:.*]] = sv.reg  : !hw.inout<i1>
    // CHECK: %[[VAL_7:.*]] = sv.read_inout %[[VAL_6]] : !hw.inout<i1>
    %q_out_reg = sv.reg  : !hw.inout<i1>
    %0 = sv.read_inout %q_out_reg : !hw.inout<i1>

    // CHECK: sv.always posedge %[[VAL_3]] {
    // CHECK:   sv.passign %[[VAL_6]], %[[VAL_5]] : i1
    // CHECK: }
    sv.always posedge %clock {
      sv.passign %q_out_reg, %d : i1
    }

    // CHECK: %[[VAL_8:.*]] = llvm.getelementptr inbounds %[[VAL_0]]{{\[}}%[[VAL_1]], 2] : (!llvm.ptr<struct<"_Struct_my_dff", packed (i1, i1, i1, i1, i1, i1)>>, i32) -> !llvm.ptr<i1>
    // CHECK: llvm.store %[[VAL_7]], %[[VAL_8]] : !llvm.ptr<i1>
    // CHECK: hw.output
    // CHECK: }
    hw.output %0 : i1
  }

// CHECK-LABEL: hw.module @Moduleshift(
// CHECK-SAME: %[[VAL_0:.*]]: !llvm.ptr<struct<"_Struct_Moduleshift", packed (i1, i1, i1, i1, ptr<struct<"_Struct_my_dff", packed (i1, i1, i1, i1, i1, i1)>>, ptr<struct<"_Struct_my_dff", packed (i1, i1, i1, i1, i1, i1)>>, ptr<struct<"_Struct_my_dff", packed (i1, i1, i1, i1, i1, i1)>>)>>) {
  hw.module @Moduleshift(%clock: i1, %reset: i1, %d: i1) -> (q: i1) {
    // CHECK: %[[VAL_1:.*]] = llvm.mlir.constant(0 : i32) : i32
    // CHECK: %[[VAL_2:.*]] = llvm.getelementptr inbounds %[[VAL_0]]{{\[}}%[[VAL_1]], 0] : (!llvm.ptr<struct<"_Struct_Moduleshift", packed (i1, i1, i1, i1, ptr<struct<"_Struct_my_dff", packed (i1, i1, i1, i1, i1, i1)>>, ptr<struct<"_Struct_my_dff", packed (i1, i1, i1, i1, i1, i1)>>, ptr<struct<"_Struct_my_dff", packed (i1, i1, i1, i1, i1, i1)>>)>>, i32) -> !llvm.ptr<i1>
    // CHECK: %[[VAL_3:.*]] = llvm.load %[[VAL_2]] : !llvm.ptr<i1>
    // CHECK: %[[VAL_4:.*]] = llvm.getelementptr inbounds %[[VAL_0]]{{\[}}%[[VAL_1]], 2] : (!llvm.ptr<struct<"_Struct_Moduleshift", packed (i1, i1, i1, i1, ptr<struct<"_Struct_my_dff", packed (i1, i1, i1, i1, i1, i1)>>, ptr<struct<"_Struct_my_dff", packed (i1, i1, i1, i1, i1, i1)>>, ptr<struct<"_Struct_my_dff", packed (i1, i1, i1, i1, i1, i1)>>)>>, i32) -> !llvm.ptr<i1>
    // CHECK: %[[VAL_5:.*]] = llvm.load %[[VAL_4]] : !llvm.ptr<i1>
    // CHECK: %[[VAL_6:.*]] = llvm.getelementptr inbounds %[[VAL_0]]{{\[}}%[[VAL_1]], 4] : (!llvm.ptr<struct<"_Struct_Moduleshift", packed (i1, i1, i1, i1, ptr<struct<"_Struct_my_dff", packed (i1, i1, i1, i1, i1, i1)>>, ptr<struct<"_Struct_my_dff", packed (i1, i1, i1, i1, i1, i1)>>, ptr<struct<"_Struct_my_dff", packed (i1, i1, i1, i1, i1, i1)>>)>>, i32) -> !llvm.ptr<ptr<struct<"_Struct_my_dff", packed (i1, i1, i1, i1, i1, i1)>>>
    // CHECK: %[[VAL_7:.*]] = llvm.load %[[VAL_6]] : !llvm.ptr<ptr<struct<"_Struct_my_dff", packed (i1, i1, i1, i1, i1, i1)>>>
    // CHECK: %[[VAL_8:.*]] = llvm.getelementptr inbounds %[[VAL_7]]{{\[}}%[[VAL_1]], 0] : (!llvm.ptr<struct<"_Struct_my_dff", packed (i1, i1, i1, i1, i1, i1)>>, i32) -> !llvm.ptr<i1>
    // CHECK: llvm.store %[[VAL_3]], %[[VAL_8]] : !llvm.ptr<i1>
    // CHECK: %[[VAL_9:.*]] = llvm.getelementptr inbounds %[[VAL_7]]{{\[}}%[[VAL_1]], 1] : (!llvm.ptr<struct<"_Struct_my_dff", packed (i1, i1, i1, i1, i1, i1)>>, i32) -> !llvm.ptr<i1>
    // CHECK: llvm.store %[[VAL_5]], %[[VAL_9]] : !llvm.ptr<i1>
    // CHECK: hw.instance "dff0" @my_dff(ptr_struct_my_dff: %[[VAL_7]]: !llvm.ptr<struct<"_Struct_my_dff", packed (i1, i1, i1, i1, i1, i1)>>) -> ()
    %dff0.q = hw.instance "dff0" @my_dff(clock: %clock: i1, d: %d: i1) -> (q: i1) {sv.namehint = "q_net_0"}

    // CHECK: %[[VAL_10:.*]] = llvm.mlir.constant(2 : i32) : i32
    // CHECK: %[[VAL_11:.*]] = llvm.getelementptr inbounds %[[VAL_7]]{{\[}}%[[VAL_1]], 2] : (!llvm.ptr<struct<"_Struct_my_dff", packed (i1, i1, i1, i1, i1, i1)>>, i32) -> !llvm.ptr<i1>
    // CHECK: %[[VAL_12:.*]] = llvm.load %[[VAL_11]] : !llvm.ptr<i1>
    // CHECK: %[[VAL_13:.*]] = llvm.getelementptr inbounds %[[VAL_0]]{{\[}}%[[VAL_1]], 5] : (!llvm.ptr<struct<"_Struct_Moduleshift", packed (i1, i1, i1, i1, ptr<struct<"_Struct_my_dff", packed (i1, i1, i1, i1, i1, i1)>>, ptr<struct<"_Struct_my_dff", packed (i1, i1, i1, i1, i1, i1)>>, ptr<struct<"_Struct_my_dff", packed (i1, i1, i1, i1, i1, i1)>>)>>, i32) -> !llvm.ptr<ptr<struct<"_Struct_my_dff", packed (i1, i1, i1, i1, i1, i1)>>>
    // CHECK: %[[VAL_14:.*]] = llvm.load %[[VAL_13]] : !llvm.ptr<ptr<struct<"_Struct_my_dff", packed (i1, i1, i1, i1, i1, i1)>>>
    // CHECK: %[[VAL_15:.*]] = llvm.getelementptr inbounds %[[VAL_14]]{{\[}}%[[VAL_1]], 0] : (!llvm.ptr<struct<"_Struct_my_dff", packed (i1, i1, i1, i1, i1, i1)>>, i32) -> !llvm.ptr<i1>
    // CHECK: llvm.store %[[VAL_3]], %[[VAL_15]] : !llvm.ptr<i1>
    // CHECK: %[[VAL_16:.*]] = llvm.getelementptr inbounds %[[VAL_14]]{{\[}}%[[VAL_1]], 1] : (!llvm.ptr<struct<"_Struct_my_dff", packed (i1, i1, i1, i1, i1, i1)>>, i32) -> !llvm.ptr<i1>
    // CHECK: llvm.store %[[VAL_12]], %[[VAL_16]] : !llvm.ptr<i1>
    // CHECK: hw.instance "dff1" @my_dff(ptr_struct_my_dff: %[[VAL_14]]: !llvm.ptr<struct<"_Struct_my_dff", packed (i1, i1, i1, i1, i1, i1)>>) -> ()
    %dff1.q = hw.instance "dff1" @my_dff(clock: %clock: i1, d: %dff0.q: i1) -> (q: i1) {sv.namehint = "q_net_1"}

    // CHECK: %[[VAL_17:.*]] = llvm.mlir.constant(2 : i32) : i32
    // CHECK: %[[VAL_18:.*]] = llvm.getelementptr inbounds %[[VAL_14]]{{\[}}%[[VAL_1]], 2] : (!llvm.ptr<struct<"_Struct_my_dff", packed (i1, i1, i1, i1, i1, i1)>>, i32) -> !llvm.ptr<i1>
    // CHECK: %[[VAL_19:.*]] = llvm.load %[[VAL_18]] : !llvm.ptr<i1>
    // CHECK: %[[VAL_20:.*]] = llvm.getelementptr inbounds %[[VAL_0]]{{\[}}%[[VAL_1]], 6] : (!llvm.ptr<struct<"_Struct_Moduleshift", packed (i1, i1, i1, i1, ptr<struct<"_Struct_my_dff", packed (i1, i1, i1, i1, i1, i1)>>, ptr<struct<"_Struct_my_dff", packed (i1, i1, i1, i1, i1, i1)>>, ptr<struct<"_Struct_my_dff", packed (i1, i1, i1, i1, i1, i1)>>)>>, i32) -> !llvm.ptr<ptr<struct<"_Struct_my_dff", packed (i1, i1, i1, i1, i1, i1)>>>
    // CHECK: %[[VAL_21:.*]] = llvm.load %[[VAL_20]] : !llvm.ptr<ptr<struct<"_Struct_my_dff", packed (i1, i1, i1, i1, i1, i1)>>>
    // CHECK: %[[VAL_22:.*]] = llvm.getelementptr inbounds %[[VAL_21]]{{\[}}%[[VAL_1]], 0] : (!llvm.ptr<struct<"_Struct_my_dff", packed (i1, i1, i1, i1, i1, i1)>>, i32) -> !llvm.ptr<i1>
    // CHECK: llvm.store %[[VAL_3]], %[[VAL_22]] : !llvm.ptr<i1>
    // CHECK: %[[VAL_23:.*]] = llvm.getelementptr inbounds %[[VAL_21]]{{\[}}%[[VAL_1]], 1] : (!llvm.ptr<struct<"_Struct_my_dff", packed (i1, i1, i1, i1, i1, i1)>>, i32) -> !llvm.ptr<i1>
    // CHECK: llvm.store %[[VAL_19]], %[[VAL_23]] : !llvm.ptr<i1>
    // CHECK: hw.instance "dff2" @my_dff(ptr_struct_my_dff: %[[VAL_21]]: !llvm.ptr<struct<"_Struct_my_dff", packed (i1, i1, i1, i1, i1, i1)>>) -> ()
    %dff2.q = hw.instance "dff2" @my_dff(clock: %clock: i1, d: %dff1.q: i1) -> (q: i1)

    // CHECK: %[[VAL_24:.*]] = llvm.mlir.constant(2 : i32) : i32
    // CHECK: %[[VAL_25:.*]] = llvm.getelementptr inbounds %[[VAL_21]]{{\[}}%[[VAL_1]], 2] : (!llvm.ptr<struct<"_Struct_my_dff", packed (i1, i1, i1, i1, i1, i1)>>, i32) -> !llvm.ptr<i1>
    // CHECK: %[[VAL_26:.*]] = llvm.load %[[VAL_25]] : !llvm.ptr<i1>
    // CHECK: %[[VAL_27:.*]] = llvm.getelementptr inbounds %[[VAL_0]]{{\[}}%[[VAL_1]], 3] : (!llvm.ptr<struct<"_Struct_Moduleshift", packed (i1, i1, i1, i1, ptr<struct<"_Struct_my_dff", packed (i1, i1, i1, i1, i1, i1)>>, ptr<struct<"_Struct_my_dff", packed (i1, i1, i1, i1, i1, i1)>>, ptr<struct<"_Struct_my_dff", packed (i1, i1, i1, i1, i1, i1)>>)>>, i32) -> !llvm.ptr<i1>
    // CHECK: llvm.store %[[VAL_26]], %[[VAL_27]] : !llvm.ptr<i1>
    // CHECK: hw.output
    // CHECK: }
    hw.output %dff2.q : i1
  }

// CHECK-LABEL: hw.module @always_inline_expr(
// CHECK-SAME: %[[VAL_0:.*]]: !llvm.ptr<struct<"_Struct_always_inline_expr", packed (i1, i1, i1, i1, i1, i1, i1, i5, i5, array<2 x i5>, i1, array<2 x i5>)>>) attributes {sv.trigger = [3 : i8]} {
  hw.module @always_inline_expr(%ro_clock_0: i1, %ro_en_0: i1, %ro_addr_0: i1, %wo_clock_0: i1, %wo_en_0: i1, %wo_addr_0: i1, %wo_mask_0: i1, %wo_data_0: i5) -> (ro_data_0: i5) attributes {sv.trigger = [3 : i8]} {
    // CHECK: %[[VAL_1:.*]] = llvm.mlir.constant(0 : i32) : i32
    // CHECK: %[[VAL_2:.*]] = llvm.getelementptr inbounds %[[VAL_0]]{{\[}}%[[VAL_1]], 1] : (!llvm.ptr<struct<"_Struct_always_inline_expr", packed (i1, i1, i1, i1, i1, i1, i1, i5, i5, array<2 x i5>, i1, array<2 x i5>)>>, i32) -> !llvm.ptr<i1>
    // CHECK: %[[VAL_3:.*]] = llvm.load %[[VAL_2]] : !llvm.ptr<i1>
    // CHECK: %[[VAL_4:.*]] = llvm.getelementptr inbounds %[[VAL_0]]{{\[}}%[[VAL_1]], 2] : (!llvm.ptr<struct<"_Struct_always_inline_expr", packed (i1, i1, i1, i1, i1, i1, i1, i5, i5, array<2 x i5>, i1, array<2 x i5>)>>, i32) -> !llvm.ptr<i1>
    // CHECK: %[[VAL_5:.*]] = llvm.load %[[VAL_4]] : !llvm.ptr<i1>
    // CHECK: %[[VAL_6:.*]] = llvm.getelementptr inbounds %[[VAL_0]]{{\[}}%[[VAL_1]], 3] : (!llvm.ptr<struct<"_Struct_always_inline_expr", packed (i1, i1, i1, i1, i1, i1, i1, i5, i5, array<2 x i5>, i1, array<2 x i5>)>>, i32) -> !llvm.ptr<i1>
    // CHECK: %[[VAL_7:.*]] = llvm.load %[[VAL_6]] : !llvm.ptr<i1>
    // CHECK: %[[VAL_8:.*]] = llvm.getelementptr inbounds %[[VAL_0]]{{\[}}%[[VAL_1]], 4] : (!llvm.ptr<struct<"_Struct_always_inline_expr", packed (i1, i1, i1, i1, i1, i1, i1, i5, i5, array<2 x i5>, i1, array<2 x i5>)>>, i32) -> !llvm.ptr<i1>
    // CHECK: %[[VAL_9:.*]] = llvm.load %[[VAL_8]] : !llvm.ptr<i1>
    // CHECK: %[[VAL_10:.*]] = llvm.getelementptr inbounds %[[VAL_0]]{{\[}}%[[VAL_1]], 5] : (!llvm.ptr<struct<"_Struct_always_inline_expr", packed (i1, i1, i1, i1, i1, i1, i1, i5, i5, array<2 x i5>, i1, array<2 x i5>)>>, i32) -> !llvm.ptr<i1>
    // CHECK: %[[VAL_11:.*]] = llvm.load %[[VAL_10]] : !llvm.ptr<i1>
    // CHECK: %[[VAL_12:.*]] = llvm.getelementptr inbounds %[[VAL_0]]{{\[}}%[[VAL_1]], 6] : (!llvm.ptr<struct<"_Struct_always_inline_expr", packed (i1, i1, i1, i1, i1, i1, i1, i5, i5, array<2 x i5>, i1, array<2 x i5>)>>, i32) -> !llvm.ptr<i1>
    // CHECK: %[[VAL_13:.*]] = llvm.load %[[VAL_12]] : !llvm.ptr<i1>
    // CHECK: %[[VAL_14:.*]] = llvm.getelementptr inbounds %[[VAL_0]]{{\[}}%[[VAL_1]], 7] : (!llvm.ptr<struct<"_Struct_always_inline_expr", packed (i1, i1, i1, i1, i1, i1, i1, i5, i5, array<2 x i5>, i1, array<2 x i5>)>>, i32) -> !llvm.ptr<i5>
    // CHECK: %[[VAL_15:.*]] = llvm.load %[[VAL_14]] : !llvm.ptr<i5>

    // CHECK: %[[VAL_16:.*]] = sv.reg  : !hw.inout<uarray<2xi5>>
    // CHECK: %[[VAL_17:.*]] = sv.array_index_inout %[[VAL_16]]{{\[}}%[[VAL_5]]] : !hw.inout<uarray<2xi5>>, i1
    // CHECK: %[[VAL_18:.*]] = sv.read_inout %[[VAL_17]] : !hw.inout<i5>
    // CHECK: %[[VAL_19:.*]] = sv.constantX : i5
    // CHECK: %[[VAL_20:.*]] = comb.mux %[[VAL_3]], %[[VAL_18]], %[[VAL_19]] : i5
    %Memory = sv.reg  : !hw.inout<uarray<2xi5>>
    %0 = sv.array_index_inout %Memory[%ro_addr_0] : !hw.inout<uarray<2xi5>>, i1
    %1 = sv.read_inout %0 : !hw.inout<i5>
    %x_i5 = sv.constantX : i5
    %2 = comb.mux %ro_en_0, %1, %x_i5 : i5

    // CHECK: sv.always posedge %[[VAL_7]] {
    // CHECK:   %[[VAL_21:.*]] = comb.and %[[VAL_9]], %[[VAL_13]] : i1
    // CHECK:   sv.if %[[VAL_21]] {
    // CHECK:     %[[VAL_22:.*]] = sv.array_index_inout %[[VAL_16]]{{\[}}%[[VAL_11]]] : !hw.inout<uarray<2xi5>>, i1
    // CHECK:     sv.passign %[[VAL_22]], %[[VAL_15]] : i5
    // CHECK:   }
    // CHECK: }
    sv.always posedge %wo_clock_0 {
      %3 = comb.and %wo_en_0, %wo_mask_0 : i1
      sv.if %3 {
        %4 = sv.array_index_inout %Memory[%wo_addr_0] : !hw.inout<uarray<2xi5>>, i1
        sv.passign %4, %wo_data_0 : i5
      }
    }

    // CHECK: %[[VAL_23:.*]] = llvm.getelementptr inbounds %[[VAL_0]]{{\[}}%[[VAL_1]], 8] : (!llvm.ptr<struct<"_Struct_always_inline_expr", packed (i1, i1, i1, i1, i1, i1, i1, i5, i5, array<2 x i5>, i1, array<2 x i5>)>>, i32) -> !llvm.ptr<i5>
    // CHECK: llvm.store %[[VAL_20]], %[[VAL_23]] : !llvm.ptr<i5>
    // CHECK: hw.output
    // CHECK: }
    hw.output %2 : i5
  }
}