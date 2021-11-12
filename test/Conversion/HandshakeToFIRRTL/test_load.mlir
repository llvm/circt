// RUN: circt-opt -lower-handshake-to-firrtl -split-input-file -verify-diagnostics %s

// CHECK-LABEL: firrtl.module @handshake_load_3ins_2outs_ui8(
// CHECK-SAME:  %arg0: !firrtl.bundle<valid: uint<1>, ready flip: uint<1>>, data: uint<64>>, %arg1: !firrtl.bundle<valid: uint<1>, ready flip: uint<1>>, data: uint<8>>, %arg2: !firrtl.bundle<valid: uint<1>, ready flip: uint<1>>>, %arg3: !firrtl.bundle<valid flip: uint<1>>, ready: uint<1>, data: flip<uint<8>>>, %arg4: !firrtl.bundle<valid flip: uint<1>>, ready: uint<1>, data: flip<uint<64>>>) {
// CHECK:   %[[IADDR_VALID:.+]] = firrtl.subfield %[[ARG0]]("valid") : (!firrtl.bundle<valid: uint<1>, ready flip: uint<1>>, data: uint<64>>) -> !firrtl.uint<1>
// CHECK:   %[[IADDR_READY:.+]] = firrtl.subfield %[[ARG0]]("ready") : (!firrtl.bundle<valid: uint<1>, ready flip: uint<1>>, data: uint<64>>) -> !firrtl.uint<1>
// CHECK:   %[[IADDR_DATA:.+]] = firrtl.subfield %[[ARG0]]("data") : (!firrtl.bundle<valid: uint<1>, ready flip: uint<1>>, data: uint<64>>) -> !firrtl.uint<64>
// CHECK:   %[[MDATA_VALID:.+]] = firrtl.subfield %[[ARG1]]("valid") : (!firrtl.bundle<valid: uint<1>, ready flip: uint<1>>, data: uint<8>>) -> !firrtl.uint<1>
// CHECK:   %[[MDATA_READY:.+]] = firrtl.subfield %[[ARG1]]("ready") : (!firrtl.bundle<valid: uint<1>, ready flip: uint<1>>, data: uint<8>>) -> !firrtl.uint<1>
// CHECK:   %[[MDATA_DATA:.+]] = firrtl.subfield %[[ARG1]]("data") : (!firrtl.bundle<valid: uint<1>, ready flip: uint<1>>, data: uint<8>>) -> !firrtl.uint<8>
// CHECK:   %[[CTRL_VALID:.+]] = firrtl.subfield %[[ARG2]]("valid") : (!firrtl.bundle<valid: uint<1>, ready flip: uint<1>>>) -> !firrtl.uint<1>
// CHECK:   %[[CTRL_READY:.+]] = firrtl.subfield %[[ARG2]]("ready") : (!firrtl.bundle<valid: uint<1>, ready flip: uint<1>>>) -> !firrtl.uint<1>
// CHECK:   %[[ODATA_VALID:.+]] = firrtl.subfield %[[ARG3]]("valid") : (!firrtl.bundle<valid flip: uint<1>>, ready: uint<1>, data: flip<uint<8>>>) -> !firrtl.uint<1>
// CHECK:   %[[ODATA_READY:.+]] = firrtl.subfield %[[ARG3]]("ready") : (!firrtl.bundle<valid flip: uint<1>>, ready: uint<1>, data: flip<uint<8>>>) -> !firrtl.uint<1>
// CHECK:   %[[ODATA_DATA:.+]] = firrtl.subfield %[[ARG3]]("data") : (!firrtl.bundle<valid flip: uint<1>>, ready: uint<1>, data: flip<uint<8>>>) -> !firrtl.uint<8>
// CHECK:   %[[MADDR_VALID:.+]] = firrtl.subfield %[[ARG4]]("valid") : (!firrtl.bundle<valid flip: uint<1>>, ready: uint<1>, data: flip<uint<64>>>) -> !firrtl.uint<1>
// CHECK:   %[[MADDR_READY:.+]] = firrtl.subfield %[[ARG4]]("ready") : (!firrtl.bundle<valid flip: uint<1>>, ready: uint<1>, data: flip<uint<64>>>) -> !firrtl.uint<1>
// CHECK:   %[[MADDR_DATA:.+]] = firrtl.subfield %[[ARG4]]("data") : (!firrtl.bundle<valid flip: uint<1>>, ready: uint<1>, data: flip<uint<64>>>) -> !firrtl.uint<64>

// CHECK:   firrtl.connect %[[MADDR_DATA:.+]], %[[IADDR_DATA:.+]] : !firrtl.uint<64>, !firrtl.uint<64>
// CHECK:   firrtl.connect %[[ODATA_DATA:.+]], %[[MDATA_DATA:.+]] : !firrtl.uint<8>, !firrtl.uint<8>

// CHECK:   %14 = firrtl.and %[[IADDR_VALID:.+]], %[[CTRL_VALID:.+]] : (!firrtl.uint<1>, !firrtl.uint<1>) -> !firrtl.uint<1>
// CHECK:   firrtl.connect %[[MADDR_VALID:.+]], %14 : !firrtl.uint<1>, !firrtl.uint<1>

// CHECK:   %15 = firrtl.and %14, %[[MADDR_READY:.+]] : (!firrtl.uint<1>, !firrtl.uint<1>) -> !firrtl.uint<1>
// CHECK:   firrtl.connect %[[IADDR_READY:.+]], %15 : !firrtl.uint<1>, !firrtl.uint<1>
// CHECK:   firrtl.connect %[[CTRL_READY:.+]], %15 : !firrtl.uint<1>, !firrtl.uint<1>

// CHECK:   firrtl.connect %[[ODATA_VALID:.+]], %[[MDATA_VALID:.+]] : !firrtl.uint<1>, !firrtl.uint<1>
// CHECK:   firrtl.connect %[[MDATA_READY:.+]], %[[ODATA_READY:.+]] : !firrtl.uint<1>, !firrtl.uint<1>
// CHECK: }

// CHECK:           firrtl.module @main(in %[[VAL_111:.*]]: !firrtl.bundle<valid: uint<1>, ready flip: uint<1>, data: uint<64>>, in %[[VAL_112:.*]]: !firrtl.bundle<valid: uint<1>, ready flip: uint<1>>, out %[[VAL_113:.*]]: !firrtl.bundle<valid: uint<1>, ready flip: uint<1>>, in %[[VAL_114:.*]]: !firrtl.clock, in %[[VAL_115:.*]]: !firrtl.uint<1>) {
// CHECK:             %[[VAL_129:.*]], %[[VAL_130:.*]], %[[VAL_131:.*]], %[[VAL_132:.*]], %[[VAL_133:.*]] = firrtl.instance handshake_load  @handshake_load_in_ui64_ui8_out_ui8_ui64(in [[ARG0:.+]]: !firrtl.bundle<valid: uint<1>, ready flip: uint<1>, data: uint<64>>, in [[ARG1:.+]]: !firrtl.bundle<valid: uint<1>, ready flip: uint<1>, data: uint<8>>, in [[ARG2:.+]]: !firrtl.bundle<valid: uint<1>, ready flip: uint<1>>, out [[ARG3:.+]]: !firrtl.bundle<valid: uint<1>, ready flip: uint<1>, data: uint<8>>, out [[ARG4:.+]]: !firrtl.bundle<valid: uint<1>, ready flip: uint<1>, data: uint<64>>)
handshake.func @main(%arg0: index, %arg1: none, ...) -> none {
  %0:2 = "handshake.memory"(%addressResults) {id = 0 : i32, ld_count = 1 : i32, lsq = false, st_count = 0 : i32, type = memref<10xi8>} : (index) -> (i8, none)
  %1:2 = "handshake.fork"(%arg1) {control = true} : (none) -> (none, none)
  %2 = "handshake.join"(%1#1, %0#1) {control = true} : (none, none) -> none
  %3, %addressResults = "handshake.load"(%arg0, %0#0, %1#0) : (index, i8, none) -> (i8, index)
  "handshake.sink"(%3) : (i8) -> ()
  handshake.return %2 : none
}
