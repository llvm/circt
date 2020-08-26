// RUN: circt-opt %s -affine-to-handshake | FileCheck %s

func @const_values () -> () {
  %0 = constant 0 : index 
  return
}

// CHECK: handshake.func @const_values(%[[ARG0:.*]]: none, ...) -> none {
// CHECK:   %[[VAL0:.*]]:2 = "handshake.fork"(%[[ARG0]]) {control = true} : (none) -> (none, none)
// CHECK:   %[[VAL1:.*]] = "handshake.constant"(%[[VAL0]]#0) {value = 0 : index} : (none) -> index
// CHECK:   "handshake.sink"(%[[VAL1]]) : (index) -> ()
// CHECK:   handshake.return %[[VAL0]]#1 : none
// CHECK: }
