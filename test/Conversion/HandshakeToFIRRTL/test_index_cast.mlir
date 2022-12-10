// RUN: circt-opt -lower-handshake-to-firrtl %s --split-input-file | FileCheck %s
  
// CHECK:      firrtl.module @arith_index_cast_in_ui64_out_ui8(in %[[ARG0:.+]]: !firrtl.bundle<valid: uint<1>, ready flip: uint<1>, data: uint<64>>, out %[[ARG1:.+]]: !firrtl.bundle<valid: uint<1>, ready flip: uint<1>, data: uint<8>>) {
// CHECK-NEXT:   %0 = firrtl.subfield %[[ARG0]][valid] : !firrtl.bundle<valid: uint<1>, ready flip: uint<1>, data: uint<64>>
// CHECK-NEXT:   %1 = firrtl.subfield %[[ARG0]][ready] : !firrtl.bundle<valid: uint<1>, ready flip: uint<1>, data: uint<64>>
// CHECK-NEXT:   %2 = firrtl.subfield %[[ARG0]][data] : !firrtl.bundle<valid: uint<1>, ready flip: uint<1>, data: uint<64>>
// CHECK-NEXT:   %3 = firrtl.subfield %[[ARG1]][valid] : !firrtl.bundle<valid: uint<1>, ready flip: uint<1>, data: uint<8>>
// CHECK-NEXT:   %4 = firrtl.subfield %[[ARG1]][ready] : !firrtl.bundle<valid: uint<1>, ready flip: uint<1>, data: uint<8>>
// CHECK-NEXT:   %5 = firrtl.subfield %[[ARG1]][data] : !firrtl.bundle<valid: uint<1>, ready flip: uint<1>, data: uint<8>>
// CHECK-NEXT:   %6 = firrtl.bits %2 7 to 0 : (!firrtl.uint<64>) -> !firrtl.uint<8>
// CHECK-NEXT:   firrtl.connect %5, %6 : !firrtl.uint<8>, !firrtl.uint<8>
// CHECK-NEXT:   firrtl.connect %3, %0 : !firrtl.uint<1>, !firrtl.uint<1>
// CHECK-NEXT:   %7 = firrtl.and %4, %0 : (!firrtl.uint<1>, !firrtl.uint<1>) -> !firrtl.uint<1>
// CHECK-NEXT:   firrtl.connect %1, %7 : !firrtl.uint<1>, !firrtl.uint<1>
// CHECK-NEXT: }

// CHECK: firrtl.module @test_index_cast(in %[[VAL_10:.*]]: !firrtl.bundle<valid: uint<1>, ready flip: uint<1>, data: uint<64>>, in %[[VAL_11:.*]]: !firrtl.bundle<valid: uint<1>, ready flip: uint<1>>, out %[[VAL_12:.*]]: !firrtl.bundle<valid: uint<1>, ready flip: uint<1>, data: uint<8>>, out %[[VAL_13:.*]]: !firrtl.bundle<valid: uint<1>, ready flip: uint<1>>, in %[[VAL_14:.*]]: !firrtl.clock, in %[[VAL_15:.*]]: !firrtl.uint<1>) {
// CHECK:   %[[VAL_16:.*]], %[[VAL_17:.*]] = firrtl.instance arith_index_cast0  @arith_index_cast_in_ui64_out_ui8(in [[ARG0:.+]]: !firrtl.bundle<valid: uint<1>, ready flip: uint<1>, data: uint<64>>, out [[ARG1:.+]]: !firrtl.bundle<valid: uint<1>, ready flip: uint<1>, data: uint<8>>)
// CHECK:   firrtl.connect %[[VAL_16]], %[[VAL_10]] : !firrtl.bundle<valid: uint<1>, ready flip: uint<1>, data: uint<64>>, !firrtl.bundle<valid: uint<1>, ready flip: uint<1>, data: uint<64>>
// CHECK:   firrtl.connect %[[VAL_12]], %[[VAL_17]] : !firrtl.bundle<valid: uint<1>, ready flip: uint<1>, data: uint<8>>, !firrtl.bundle<valid: uint<1>, ready flip: uint<1>, data: uint<8>>
// CHECK:   firrtl.connect %[[VAL_13]], %[[VAL_11]] : !firrtl.bundle<valid: uint<1>, ready flip: uint<1>>, !firrtl.bundle<valid: uint<1>, ready flip: uint<1>>
// CHECK: }
handshake.func @test_index_cast(%arg0: index, %arg1: none, ...) -> (i8, none) {
  %0 = arith.index_cast %arg0 : index to i8
  return %0, %arg1 : i8, none
}

// -----

// CHECK:      firrtl.module @arith_index_cast_in_ui8_out_ui64(in %[[ARG0:.+]]: !firrtl.bundle<valid: uint<1>, ready flip: uint<1>, data: uint<8>>, out %[[ARG1:.+]]: !firrtl.bundle<valid: uint<1>, ready flip: uint<1>, data: uint<64>>) {
// CHECK-NEXT:   %0 = firrtl.subfield %[[ARG0]][valid] : !firrtl.bundle<valid: uint<1>, ready flip: uint<1>, data: uint<8>>
// CHECK-NEXT:   %1 = firrtl.subfield %[[ARG0]][ready] : !firrtl.bundle<valid: uint<1>, ready flip: uint<1>, data: uint<8>>
// CHECK-NEXT:   %2 = firrtl.subfield %[[ARG0]][data] : !firrtl.bundle<valid: uint<1>, ready flip: uint<1>, data: uint<8>>
// CHECK-NEXT:   %3 = firrtl.subfield %[[ARG1]][valid] : !firrtl.bundle<valid: uint<1>, ready flip: uint<1>, data: uint<64>>
// CHECK-NEXT:   %4 = firrtl.subfield %[[ARG1]][ready] : !firrtl.bundle<valid: uint<1>, ready flip: uint<1>, data: uint<64>>
// CHECK-NEXT:   %5 = firrtl.subfield %[[ARG1]][data] : !firrtl.bundle<valid: uint<1>, ready flip: uint<1>, data: uint<64>>
// CHECK-NEXT:   %6 = firrtl.pad %2, 64 : (!firrtl.uint<8>) -> !firrtl.uint<64>
// CHECK-NEXT:   firrtl.connect %5, %6 : !firrtl.uint<64>, !firrtl.uint<64>
// CHECK-NEXT:   firrtl.connect %3, %0 : !firrtl.uint<1>, !firrtl.uint<1>
// CHECK-NEXT:   %7 = firrtl.and %4, %0 : (!firrtl.uint<1>, !firrtl.uint<1>) -> !firrtl.uint<1>
// CHECK-NEXT:   firrtl.connect %1, %7 : !firrtl.uint<1>, !firrtl.uint<1>
// CHECK-NEXT: }
// CHECK-NEXT: firrtl.module @arith_index_cast_in_ui9_out_ui64(in %[[ARG0:.+]]: !firrtl.bundle<valid: uint<1>, ready flip: uint<1>, data: uint<9>>, out %[[ARG1:.+]]: !firrtl.bundle<valid: uint<1>, ready flip: uint<1>, data: uint<64>>) {
// CHECK-NEXT:   %0 = firrtl.subfield %[[ARG0]][valid] : !firrtl.bundle<valid: uint<1>, ready flip: uint<1>, data: uint<9>>
// CHECK-NEXT:   %1 = firrtl.subfield %[[ARG0]][ready] : !firrtl.bundle<valid: uint<1>, ready flip: uint<1>, data: uint<9>>
// CHECK-NEXT:   %2 = firrtl.subfield %[[ARG0]][data] : !firrtl.bundle<valid: uint<1>, ready flip: uint<1>, data: uint<9>>
// CHECK-NEXT:   %3 = firrtl.subfield %[[ARG1]][valid] : !firrtl.bundle<valid: uint<1>, ready flip: uint<1>, data: uint<64>>
// CHECK-NEXT:   %4 = firrtl.subfield %[[ARG1]][ready] : !firrtl.bundle<valid: uint<1>, ready flip: uint<1>, data: uint<64>>
// CHECK-NEXT:   %5 = firrtl.subfield %[[ARG1]][data] : !firrtl.bundle<valid: uint<1>, ready flip: uint<1>, data: uint<64>>
// CHECK-NEXT:   %6 = firrtl.pad %2, 64 : (!firrtl.uint<9>) -> !firrtl.uint<64>
// CHECK-NEXT:   firrtl.connect %5, %6 : !firrtl.uint<64>, !firrtl.uint<64>
// CHECK-NEXT:   firrtl.connect %3, %0 : !firrtl.uint<1>, !firrtl.uint<1>
// CHECK-NEXT:   %7 = firrtl.and %4, %0 : (!firrtl.uint<1>, !firrtl.uint<1>) -> !firrtl.uint<1>
// CHECK-NEXT:   firrtl.connect %1, %7 : !firrtl.uint<1>, !firrtl.uint<1>
// CHECK-NEXT: }

// CHECK: firrtl.module @test_index_cast2(in %[[VAL_20:.*]]: !firrtl.bundle<valid: uint<1>, ready flip: uint<1>, data: uint<8>>, in %[[VAL_21:.*]]: !firrtl.bundle<valid: uint<1>, ready flip: uint<1>, data: uint<9>>, in %[[VAL_22:.*]]: !firrtl.bundle<valid: uint<1>, ready flip: uint<1>>, out %[[VAL_23:.*]]: !firrtl.bundle<valid: uint<1>, ready flip: uint<1>, data: uint<64>>, out %[[VAL_24:.*]]: !firrtl.bundle<valid: uint<1>, ready flip: uint<1>, data: uint<64>>, out %[[VAL_25:.*]]: !firrtl.bundle<valid: uint<1>, ready flip: uint<1>>, in %[[VAL_26:.*]]: !firrtl.clock, in %[[VAL_27:.*]]: !firrtl.uint<1>) {
// CHECK:   %[[VAL_28:.*]], %[[VAL_29:.*]] = firrtl.instance arith_index_cast0  @arith_index_cast_in_ui8_out_ui64(in [[ARG0:.+]]: !firrtl.bundle<valid: uint<1>, ready flip: uint<1>, data: uint<8>>, out [[ARG1:.+]]: !firrtl.bundle<valid: uint<1>, ready flip: uint<1>, data: uint<64>>)
// CHECK:   %[[VAL_30:.*]], %[[VAL_31:.*]] = firrtl.instance arith_index_cast1  @arith_index_cast_in_ui9_out_ui64(in [[ARG0:.+]]: !firrtl.bundle<valid: uint<1>, ready flip: uint<1>, data: uint<9>>, out [[ARG1:.+]]: !firrtl.bundle<valid: uint<1>, ready flip: uint<1>, data: uint<64>>)
// CHECK:   firrtl.connect %[[VAL_28]], %[[VAL_20]] : !firrtl.bundle<valid: uint<1>, ready flip: uint<1>, data: uint<8>>, !firrtl.bundle<valid: uint<1>, ready flip: uint<1>, data: uint<8>>
// CHECK:   firrtl.connect %[[VAL_30]], %[[VAL_21]] : !firrtl.bundle<valid: uint<1>, ready flip: uint<1>, data: uint<9>>, !firrtl.bundle<valid: uint<1>, ready flip: uint<1>, data: uint<9>>
// CHECK:   firrtl.connect %[[VAL_23]], %[[VAL_29]] : !firrtl.bundle<valid: uint<1>, ready flip: uint<1>, data: uint<64>>, !firrtl.bundle<valid: uint<1>, ready flip: uint<1>, data: uint<64>>
// CHECK:   firrtl.connect %[[VAL_24]], %[[VAL_31]] : !firrtl.bundle<valid: uint<1>, ready flip: uint<1>, data: uint<64>>, !firrtl.bundle<valid: uint<1>, ready flip: uint<1>, data: uint<64>>
// CHECK:   firrtl.connect %[[VAL_25]], %[[VAL_22]] : !firrtl.bundle<valid: uint<1>, ready flip: uint<1>>, !firrtl.bundle<valid: uint<1>, ready flip: uint<1>>
// CHECK: }
handshake.func @test_index_cast2(%arg0: i8, %arg1 : i9, %arg2: none, ...) -> (index, index, none) {
  %0 = arith.index_cast %arg0 : i8 to index
  %1 = arith.index_cast %arg1 : i9 to index
  return %0, %1, %arg2 : index, index, none
}
