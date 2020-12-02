// RUN: circt-opt -lower-handshake-to-firrtl %s | FileCheck %s

// CHECK-LABEL: firrtl.module @handshake_merge_1ins_1outs(
// CHECK-SAME:  %arg0: !firrtl.bundle<valid: uint<1>, ready: flip<uint<1>>, data: uint<64>>, %arg1: !firrtl.bundle<valid: flip<uint<1>>, ready: uint<1>, data: flip<uint<64>>>) {
// CHECK:   %[[ARG0_VALID:.+]] = firrtl.subfield %arg0("valid") : (!firrtl.bundle<valid: uint<1>, ready: flip<uint<1>>, data: uint<64>>) -> !firrtl.uint<1>
// CHECK:   %[[ARG0_READY:.+]] = firrtl.subfield %arg0("ready") : (!firrtl.bundle<valid: uint<1>, ready: flip<uint<1>>, data: uint<64>>) -> !firrtl.flip<uint<1>>
// CHECK:   %[[ARG0_DATA:.+]] = firrtl.subfield %arg0("data") : (!firrtl.bundle<valid: uint<1>, ready: flip<uint<1>>, data: uint<64>>) -> !firrtl.uint<64>
// CHECK:   %[[ARG1_VALID:.+]] = firrtl.subfield %arg1("valid") : (!firrtl.bundle<valid: flip<uint<1>>, ready: uint<1>, data: flip<uint<64>>>) -> !firrtl.flip<uint<1>>
// CHECK:   %[[ARG1_READY:.+]] = firrtl.subfield %arg1("ready") : (!firrtl.bundle<valid: flip<uint<1>>, ready: uint<1>, data: flip<uint<64>>>) -> !firrtl.uint<1>
// CHECK:   %[[ARG1_DATA:.+]] = firrtl.subfield %arg1("data") : (!firrtl.bundle<valid: flip<uint<1>>, ready: uint<1>, data: flip<uint<64>>>) -> !firrtl.flip<uint<64>>
// CHECK:   %[[NO_WINNER:.+]] = firrtl.constant(-1 : si2) : !firrtl.sint<2>
// CHECK:   %[[WIN:win]] = firrtl.wire {{.*}} : !firrtl.sint<2>
// CHECK:   %[[RESULT_DONE:resultDone]] = firrtl.wire {{.*}} : !firrtl.uint<1>
// CHECK:   %[[HAS_WINNER:.+]] = firrtl.neq %[[WIN]], %[[NO_WINNER]]
// CHECK:   %[[INDEX0:.+]] = firrtl.constant(0 : si2)
// CHECK:   %[[ARB0:.+]] = firrtl.mux(%[[ARG0_VALID]], %[[INDEX0]], %[[NO_WINNER]])
// CHECK:   firrtl.connect %[[WIN]], %[[ARB0]]
// CHECK:   %[[WIN_BITS:.+]] = firrtl.bits %[[WIN]] {{.+}} to 0 {{.+}} -> !firrtl.uint
// CHECK:   %[[WIN_UNSIGNED:.+]] = firrtl.asUInt %[[WIN_BITS]]
// CHECK:   %[[BITS0:.+]] = firrtl.bits %[[WIN_UNSIGNED]] 0 to 0
// CHECK:   %[[RESULT_VALID:.+]] = firrtl.and %[[ARG0_VALID]], %[[HAS_WINNER]]
// CHECK:   firrtl.connect %[[ARG1_VALID]], %[[RESULT_VALID]]
// CHECK:   %[[BITS1:.+]] = firrtl.bits %[[WIN_UNSIGNED]] 0 to 0
// CHECK:   firrtl.connect %[[ARG1_DATA]], %[[ARG0_DATA]]
// CHECK:   %[[RESULT_DONE0:.+]] = firrtl.and %[[RESULT_VALID]], %[[ARG1_READY]]
// CHECK:   firrtl.connect %[[RESULT_DONE]], %[[RESULT_DONE0]]
// CHECK:   %[[ARG0_READY0:.+]] = firrtl.eq %[[WIN]], %[[INDEX0]]
// CHECK:   %[[ARG0_READY1:.+]] = firrtl.and %[[ARG0_READY0]], %[[RESULT_DONE]]
// CHECK:   firrtl.connect %[[ARG0_READY]], %[[ARG0_READY1]]

// CHECK-LABEL: firrtl.module @std_addi_2ins_1outs(
// CHECK-SAME:  %arg0: !firrtl.bundle<valid: uint<1>, ready: flip<uint<1>>, data: uint<64>>, %arg1: !firrtl.bundle<valid: uint<1>, ready: flip<uint<1>>, data: uint<64>>, %arg2: !firrtl.bundle<valid: flip<uint<1>>, ready: uint<1>, data: flip<uint<64>>>) {
// CHECK:   %0 = firrtl.subfield %arg0("valid") : (!firrtl.bundle<valid: uint<1>, ready: flip<uint<1>>, data: uint<64>>) -> !firrtl.uint<1>
// CHECK:   %1 = firrtl.subfield %arg0("ready") : (!firrtl.bundle<valid: uint<1>, ready: flip<uint<1>>, data: uint<64>>) -> !firrtl.flip<uint<1>>
// CHECK:   %2 = firrtl.subfield %arg0("data") : (!firrtl.bundle<valid: uint<1>, ready: flip<uint<1>>, data: uint<64>>) -> !firrtl.uint<64>
// CHECK:   %3 = firrtl.subfield %arg1("valid") : (!firrtl.bundle<valid: uint<1>, ready: flip<uint<1>>, data: uint<64>>) -> !firrtl.uint<1>
// CHECK:   %4 = firrtl.subfield %arg1("ready") : (!firrtl.bundle<valid: uint<1>, ready: flip<uint<1>>, data: uint<64>>) -> !firrtl.flip<uint<1>>
// CHECK:   %5 = firrtl.subfield %arg1("data") : (!firrtl.bundle<valid: uint<1>, ready: flip<uint<1>>, data: uint<64>>) -> !firrtl.uint<64>
// CHECK:   %6 = firrtl.subfield %arg2("valid") : (!firrtl.bundle<valid: flip<uint<1>>, ready: uint<1>, data: flip<uint<64>>>) -> !firrtl.flip<uint<1>>
// CHECK:   %7 = firrtl.subfield %arg2("ready") : (!firrtl.bundle<valid: flip<uint<1>>, ready: uint<1>, data: flip<uint<64>>>) -> !firrtl.uint<1>
// CHECK:   %8 = firrtl.subfield %arg2("data") : (!firrtl.bundle<valid: flip<uint<1>>, ready: uint<1>, data: flip<uint<64>>>) -> !firrtl.flip<uint<64>>
// CHECK:   %9 = firrtl.add %2, %5 : (!firrtl.uint<64>, !firrtl.uint<64>) -> !firrtl.uint<64>
// CHECK:   firrtl.connect %8, %9 : !firrtl.flip<uint<64>>, !firrtl.uint<64>
// CHECK:   %10 = firrtl.and %0, %3 : (!firrtl.uint<1>, !firrtl.uint<1>) -> !firrtl.uint<1>
// CHECK:   firrtl.connect %6, %10 : !firrtl.flip<uint<1>>, !firrtl.uint<1>
// CHECK:   %11 = firrtl.and %7, %10 : (!firrtl.uint<1>, !firrtl.uint<1>) -> !firrtl.uint<1>
// CHECK:   firrtl.connect %1, %11 : !firrtl.flip<uint<1>>, !firrtl.uint<1>
// CHECK:   firrtl.connect %4, %11 : !firrtl.flip<uint<1>>, !firrtl.uint<1>
// CHECK: }

// CHECK-LABEL: firrtl.module @simple_addi(
// CHECK-SAME:  %arg0: !firrtl.bundle<valid: uint<1>, ready: flip<uint<1>>, data: uint<64>>, %arg1: !firrtl.bundle<valid: uint<1>, ready: flip<uint<1>>, data: uint<64>>, %arg2: !firrtl.bundle<valid: uint<1>, ready: flip<uint<1>>>, %arg3: !firrtl.bundle<valid: flip<uint<1>>, ready: uint<1>, data: flip<uint<64>>>, %arg4: !firrtl.bundle<valid: flip<uint<1>>, ready: uint<1>>, %clock: !firrtl.clock, %reset: !firrtl.uint<1>) {
handshake.func @simple_addi(%arg0: index, %arg1: index, %arg2: none, ...) -> (index, none) {

  // CHECK: %0 = firrtl.instance @handshake_merge_1ins_1outs {name = ""} : !firrtl.bundle<arg0: bundle<valid: flip<uint<1>>, ready: uint<1>, data: flip<uint<64>>>, arg1: bundle<valid: uint<1>, ready: flip<uint<1>>, data: uint<64>>>
  // CHECK: %1 = firrtl.subfield %0("arg0") : (!firrtl.bundle<arg0: bundle<valid: flip<uint<1>>, ready: uint<1>, data: flip<uint<64>>>, arg1: bundle<valid: uint<1>, ready: flip<uint<1>>, data: uint<64>>>) -> !firrtl.bundle<valid: flip<uint<1>>, ready: uint<1>, data: flip<uint<64>>>
  // CHECK: firrtl.connect %1, %arg0 : !firrtl.bundle<valid: flip<uint<1>>, ready: uint<1>, data: flip<uint<64>>>, !firrtl.bundle<valid: uint<1>, ready: flip<uint<1>>, data: uint<64>>
  // CHECK: %2 = firrtl.subfield %0("arg1") : (!firrtl.bundle<arg0: bundle<valid: flip<uint<1>>, ready: uint<1>, data: flip<uint<64>>>, arg1: bundle<valid: uint<1>, ready: flip<uint<1>>, data: uint<64>>>) -> !firrtl.bundle<valid: uint<1>, ready: flip<uint<1>>, data: uint<64>>
  %0 = "handshake.merge"(%arg0) : (index) -> index
  
  // CHECK: %3 = firrtl.instance @handshake_merge_1ins_1outs {name = ""} : !firrtl.bundle<arg0: bundle<valid: flip<uint<1>>, ready: uint<1>, data: flip<uint<64>>>, arg1: bundle<valid: uint<1>, ready: flip<uint<1>>, data: uint<64>>>
  // CHECK: %4 = firrtl.subfield %3("arg0") : (!firrtl.bundle<arg0: bundle<valid: flip<uint<1>>, ready: uint<1>, data: flip<uint<64>>>, arg1: bundle<valid: uint<1>, ready: flip<uint<1>>, data: uint<64>>>) -> !firrtl.bundle<valid: flip<uint<1>>, ready: uint<1>, data: flip<uint<64>>>
  // CHECK: firrtl.connect %4, %arg1 : !firrtl.bundle<valid: flip<uint<1>>, ready: uint<1>, data: flip<uint<64>>>, !firrtl.bundle<valid: uint<1>, ready: flip<uint<1>>, data: uint<64>>
  // CHECK: %5 = firrtl.subfield %3("arg1") : (!firrtl.bundle<arg0: bundle<valid: flip<uint<1>>, ready: uint<1>, data: flip<uint<64>>>, arg1: bundle<valid: uint<1>, ready: flip<uint<1>>, data: uint<64>>>) -> !firrtl.bundle<valid: uint<1>, ready: flip<uint<1>>, data: uint<64>>
  %1 = "handshake.merge"(%arg1) : (index) -> index
  
  // CHECK: %6 = firrtl.instance @std_addi_2ins_1outs {name = ""} : !firrtl.bundle<arg0: bundle<valid: flip<uint<1>>, ready: uint<1>, data: flip<uint<64>>>, arg1: bundle<valid: flip<uint<1>>, ready: uint<1>, data: flip<uint<64>>>, arg2: bundle<valid: uint<1>, ready: flip<uint<1>>, data: uint<64>>>
  // CHECK: %7 = firrtl.subfield %6("arg0") : (!firrtl.bundle<arg0: bundle<valid: flip<uint<1>>, ready: uint<1>, data: flip<uint<64>>>, arg1: bundle<valid: flip<uint<1>>, ready: uint<1>, data: flip<uint<64>>>, arg2: bundle<valid: uint<1>, ready: flip<uint<1>>, data: uint<64>>>) -> !firrtl.bundle<valid: flip<uint<1>>, ready: uint<1>, data: flip<uint<64>>>
  // CHECK: firrtl.connect %7, %2 : !firrtl.bundle<valid: flip<uint<1>>, ready: uint<1>, data: flip<uint<64>>>, !firrtl.bundle<valid: uint<1>, ready: flip<uint<1>>, data: uint<64>>
  // CHECK: %8 = firrtl.subfield %6("arg1") : (!firrtl.bundle<arg0: bundle<valid: flip<uint<1>>, ready: uint<1>, data: flip<uint<64>>>, arg1: bundle<valid: flip<uint<1>>, ready: uint<1>, data: flip<uint<64>>>, arg2: bundle<valid: uint<1>, ready: flip<uint<1>>, data: uint<64>>>) -> !firrtl.bundle<valid: flip<uint<1>>, ready: uint<1>, data: flip<uint<64>>>
  // CHECK: firrtl.connect %8, %5 : !firrtl.bundle<valid: flip<uint<1>>, ready: uint<1>, data: flip<uint<64>>>, !firrtl.bundle<valid: uint<1>, ready: flip<uint<1>>, data: uint<64>>
  // CHECK: %9 = firrtl.subfield %6("arg2") : (!firrtl.bundle<arg0: bundle<valid: flip<uint<1>>, ready: uint<1>, data: flip<uint<64>>>, arg1: bundle<valid: flip<uint<1>>, ready: uint<1>, data: flip<uint<64>>>, arg2: bundle<valid: uint<1>, ready: flip<uint<1>>, data: uint<64>>>) -> !firrtl.bundle<valid: uint<1>, ready: flip<uint<1>>, data: uint<64>>
  %2 = addi %0, %1 : index

  // CHECK: firrtl.connect %arg3, %9 : !firrtl.bundle<valid: flip<uint<1>>, ready: uint<1>, data: flip<uint<64>>>, !firrtl.bundle<valid: uint<1>, ready: flip<uint<1>>, data: uint<64>>
  // CHECK: firrtl.connect %arg4, %arg2 : !firrtl.bundle<valid: flip<uint<1>>, ready: uint<1>>, !firrtl.bundle<valid: uint<1>, ready: flip<uint<1>>>
  handshake.return %2, %arg2 : index, none
}
