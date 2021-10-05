// RUN: circt-opt -lower-handshake-to-firrtl %s --split-input-file | FileCheck %s
  
// CHECK:      firrtl.module @std_index_cast_in_ui64_out_ui8(in %arg0: !firrtl.bundle<valid: uint<1>, ready flip: uint<1>, data: uint<64>>, out %arg1: !firrtl.bundle<valid: uint<1>, ready flip: uint<1>, data: uint<8>>) {
// CHECK-NEXT:   %0 = firrtl.subfield %arg0(0) : (!firrtl.bundle<valid: uint<1>, ready flip: uint<1>, data: uint<64>>) -> !firrtl.uint<1>
// CHECK-NEXT:   %1 = firrtl.subfield %arg0(1) : (!firrtl.bundle<valid: uint<1>, ready flip: uint<1>, data: uint<64>>) -> !firrtl.uint<1>
// CHECK-NEXT:   %2 = firrtl.subfield %arg0(2) : (!firrtl.bundle<valid: uint<1>, ready flip: uint<1>, data: uint<64>>) -> !firrtl.uint<64>
// CHECK-NEXT:   %3 = firrtl.subfield %arg1(0) : (!firrtl.bundle<valid: uint<1>, ready flip: uint<1>, data: uint<8>>) -> !firrtl.uint<1>
// CHECK-NEXT:   %4 = firrtl.subfield %arg1(1) : (!firrtl.bundle<valid: uint<1>, ready flip: uint<1>, data: uint<8>>) -> !firrtl.uint<1>
// CHECK-NEXT:   %5 = firrtl.subfield %arg1(2) : (!firrtl.bundle<valid: uint<1>, ready flip: uint<1>, data: uint<8>>) -> !firrtl.uint<8>
// CHECK-NEXT:   %6 = firrtl.bits %2 7 to 0 : (!firrtl.uint<64>) -> !firrtl.uint<8>
// CHECK-NEXT:   firrtl.connect %5, %6 : !firrtl.uint<8>, !firrtl.uint<8>
// CHECK-NEXT:   firrtl.connect %3, %0 : !firrtl.uint<1>, !firrtl.uint<1>
// CHECK-NEXT:   %7 = firrtl.and %4, %0 : (!firrtl.uint<1>, !firrtl.uint<1>) -> !firrtl.uint<1>
// CHECK-NEXT:   firrtl.connect %1, %7 : !firrtl.uint<1>, !firrtl.uint<1>
// CHECK-NEXT: }

// CHECK-LABEL: firrtl.module @test_index_cast(
// CHECK-SAME:  in %arg0: !firrtl.bundle<valid: uint<1>, ready flip: uint<1>, data: uint<64>>,
// CHECK-SAME:  in %arg1: !firrtl.bundle<valid: uint<1>, ready flip: uint<1>>,
// CHECK-SAME:  out %arg2: !firrtl.bundle<valid: uint<1>, ready flip: uint<1>, data: uint<8>>,
// CHECK-SAME:  out %arg3: !firrtl.bundle<valid: uint<1>, ready flip: uint<1>>,
// CHECK-SAME:  in %clock: !firrtl.clock, in %reset: !firrtl.uint<1>) {
handshake.func @test_index_cast(%arg0: index, %arg1: none, ...) -> (i8, none) {
  // CHECK:      %inst_arg0, %inst_arg1 = firrtl.instance @std_index_cast_in_ui64_out_ui8  {name = ""} : !firrtl.bundle<valid: uint<1>, ready flip: uint<1>, data: uint<64>>, !firrtl.bundle<valid: uint<1>, ready flip: uint<1>, data: uint<8>>
  // CHECK-NEXT: firrtl.connect %inst_arg0, %arg0 : !firrtl.bundle<valid: uint<1>, ready flip: uint<1>, data: uint<64>>, !firrtl.bundle<valid: uint<1>, ready flip: uint<1>, data: uint<64>>
  %0 = index_cast %arg0 : index to i8

  // CHECK:      firrtl.connect %arg2, %inst_arg1 : !firrtl.bundle<valid: uint<1>, ready flip: uint<1>, data: uint<8>>, !firrtl.bundle<valid: uint<1>, ready flip: uint<1>, data: uint<8>>
  // CHECK-NEXT: firrtl.connect %arg3, %arg1 : !firrtl.bundle<valid: uint<1>, ready flip: uint<1>>, !firrtl.bundle<valid: uint<1>, ready flip: uint<1>>
  handshake.return %0, %arg1 : i8, none
}

// -----

// CHECK:      firrtl.module @std_index_cast_in_ui8_out_ui64(in %arg0: !firrtl.bundle<valid: uint<1>, ready flip: uint<1>, data: uint<8>>, out %arg1: !firrtl.bundle<valid: uint<1>, ready flip: uint<1>, data: uint<64>>) {
// CHECK-NEXT:   %0 = firrtl.subfield %arg0(0) : (!firrtl.bundle<valid: uint<1>, ready flip: uint<1>, data: uint<8>>) -> !firrtl.uint<1>
// CHECK-NEXT:   %1 = firrtl.subfield %arg0(1) : (!firrtl.bundle<valid: uint<1>, ready flip: uint<1>, data: uint<8>>) -> !firrtl.uint<1>
// CHECK-NEXT:   %2 = firrtl.subfield %arg0(2) : (!firrtl.bundle<valid: uint<1>, ready flip: uint<1>, data: uint<8>>) -> !firrtl.uint<8>
// CHECK-NEXT:   %3 = firrtl.subfield %arg1(0) : (!firrtl.bundle<valid: uint<1>, ready flip: uint<1>, data: uint<64>>) -> !firrtl.uint<1>
// CHECK-NEXT:   %4 = firrtl.subfield %arg1(1) : (!firrtl.bundle<valid: uint<1>, ready flip: uint<1>, data: uint<64>>) -> !firrtl.uint<1>
// CHECK-NEXT:   %5 = firrtl.subfield %arg1(2) : (!firrtl.bundle<valid: uint<1>, ready flip: uint<1>, data: uint<64>>) -> !firrtl.uint<64>
// CHECK-NEXT:   %6 = firrtl.pad %2, 64 : (!firrtl.uint<8>) -> !firrtl.uint<64>
// CHECK-NEXT:   firrtl.connect %5, %6 : !firrtl.uint<64>, !firrtl.uint<64>
// CHECK-NEXT:   firrtl.connect %3, %0 : !firrtl.uint<1>, !firrtl.uint<1>
// CHECK-NEXT:   %7 = firrtl.and %4, %0 : (!firrtl.uint<1>, !firrtl.uint<1>) -> !firrtl.uint<1>
// CHECK-NEXT:   firrtl.connect %1, %7 : !firrtl.uint<1>, !firrtl.uint<1>
// CHECK-NEXT: }
// CHECK-NEXT: firrtl.module @std_index_cast_in_ui9_out_ui64(in %arg0: !firrtl.bundle<valid: uint<1>, ready flip: uint<1>, data: uint<9>>, out %arg1: !firrtl.bundle<valid: uint<1>, ready flip: uint<1>, data: uint<64>>) {
// CHECK-NEXT:   %0 = firrtl.subfield %arg0(0) : (!firrtl.bundle<valid: uint<1>, ready flip: uint<1>, data: uint<9>>) -> !firrtl.uint<1>
// CHECK-NEXT:   %1 = firrtl.subfield %arg0(1) : (!firrtl.bundle<valid: uint<1>, ready flip: uint<1>, data: uint<9>>) -> !firrtl.uint<1>
// CHECK-NEXT:   %2 = firrtl.subfield %arg0(2) : (!firrtl.bundle<valid: uint<1>, ready flip: uint<1>, data: uint<9>>) -> !firrtl.uint<9>
// CHECK-NEXT:   %3 = firrtl.subfield %arg1(0) : (!firrtl.bundle<valid: uint<1>, ready flip: uint<1>, data: uint<64>>) -> !firrtl.uint<1>
// CHECK-NEXT:   %4 = firrtl.subfield %arg1(1) : (!firrtl.bundle<valid: uint<1>, ready flip: uint<1>, data: uint<64>>) -> !firrtl.uint<1>
// CHECK-NEXT:   %5 = firrtl.subfield %arg1(2) : (!firrtl.bundle<valid: uint<1>, ready flip: uint<1>, data: uint<64>>) -> !firrtl.uint<64>
// CHECK-NEXT:   %6 = firrtl.pad %2, 64 : (!firrtl.uint<9>) -> !firrtl.uint<64>
// CHECK-NEXT:   firrtl.connect %5, %6 : !firrtl.uint<64>, !firrtl.uint<64>
// CHECK-NEXT:   firrtl.connect %3, %0 : !firrtl.uint<1>, !firrtl.uint<1>
// CHECK-NEXT:   %7 = firrtl.and %4, %0 : (!firrtl.uint<1>, !firrtl.uint<1>) -> !firrtl.uint<1>
// CHECK-NEXT:   firrtl.connect %1, %7 : !firrtl.uint<1>, !firrtl.uint<1>
// CHECK-NEXT: }

// CHECK-LABEL: firrtl.module @test_index_cast2(
// CHECK-SAME:  in %arg0: !firrtl.bundle<valid: uint<1>, ready flip: uint<1>, data: uint<8>>, 
// CHECK-SAME:  in %arg1: !firrtl.bundle<valid: uint<1>, ready flip: uint<1>, data: uint<9>>, 
// CHECK-SAME:  in %arg2: !firrtl.bundle<valid: uint<1>, ready flip: uint<1>>, 
// CHECK-SAME:  out %arg3: !firrtl.bundle<valid: uint<1>, ready flip: uint<1>, data: uint<64>>, 
// CHECK-SAME:  out %arg4: !firrtl.bundle<valid: uint<1>, ready flip: uint<1>, data: uint<64>>, 
// CHECK-SAME:  out %arg5: !firrtl.bundle<valid: uint<1>, ready flip: uint<1>>, 
// CHECK-SAME:  in %clock: !firrtl.clock, in %reset: !firrtl.uint<1>) {
handshake.func @test_index_cast2(%arg0: i8, %arg1 : i9, %arg2: none, ...) -> (index, index, none) {
// CHECK: %inst_arg0, %inst_arg1 = firrtl.instance @std_index_cast_in_ui8_out_ui64  {name = ""} : !firrtl.bundle<valid: uint<1>, ready flip: uint<1>, data: uint<8>>, !firrtl.bundle<valid: uint<1>, ready flip: uint<1>, data: uint<64>>
// CHECK: %inst_arg0_0, %inst_arg1_1 = firrtl.instance @std_index_cast_in_ui9_out_ui64  {name = ""} : !firrtl.bundle<valid: uint<1>, ready flip: uint<1>, data: uint<9>>, !firrtl.bundle<valid: uint<1>, ready flip: uint<1>, data: uint<64>>
// CHECK: firrtl.connect %inst_arg0, %arg0 : !firrtl.bundle<valid: uint<1>, ready flip: uint<1>, data: uint<8>>, !firrtl.bundle<valid: uint<1>, ready flip: uint<1>, data: uint<8>>
// CHECK: firrtl.connect %inst_arg0_0, %arg1 : !firrtl.bundle<valid: uint<1>, ready flip: uint<1>, data: uint<9>>, !firrtl.bundle<valid: uint<1>, ready flip: uint<1>, data: uint<9>>
  %0 = index_cast %arg0 : i8 to index
  %1 = index_cast %arg1 : i9 to index

// CHECK:      firrtl.connect %arg3, %inst_arg1 : !firrtl.bundle<valid: uint<1>, ready flip: uint<1>, data: uint<64>>, !firrtl.bundle<valid: uint<1>, ready flip: uint<1>, data: uint<64>>
// CHECK-NEXT: firrtl.connect %arg4, %inst_arg1_1 : !firrtl.bundle<valid: uint<1>, ready flip: uint<1>, data: uint<64>>, !firrtl.bundle<valid: uint<1>, ready flip: uint<1>, data: uint<64>>
// CHECK-NEXT: firrtl.connect %arg5, %arg2 : !firrtl.bundle<valid: uint<1>, ready flip: uint<1>>, !firrtl.bundle<valid: uint<1>, ready flip: uint<1>>
  handshake.return %0, %1, %arg2 : index, index, none
}
