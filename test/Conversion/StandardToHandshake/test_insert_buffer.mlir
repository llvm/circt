// RUN: circt-opt -handshake-insert-buffer %s | FileCheck %s
module {
  handshake.func @simple_loop(%arg0: none, ...) -> none {
    // CHECK: handshake.buffer
    %0 = "handshake.branch"(%arg0) {control = true} : (none) -> none
    // CHECK: handshake.buffer
    %1:2 = "handshake.control_merge"(%0) {control = true} : (none) -> (none, index)
    // CHECK: handshake.buffer
    %2:3 = "handshake.fork"(%1#0) {control = true} : (none) -> (none, none, none)
    // CHECK: handshake.buffer
    "handshake.sink"(%1#1) : (index) -> ()
    // CHECK: handshake.buffer
    %3 = "handshake.constant"(%2#1) {value = 1 : index} : (none) -> index
    // CHECK: handshake.buffer
    %4 = "handshake.constant"(%2#0) {value = 42 : index} : (none) -> index
    // CHECK: handshake.buffer
    %5 = "handshake.branch"(%2#2) {control = true} : (none) -> none
    // CHECK: handshake.buffer
    %6 = "handshake.branch"(%3) {control = false} : (index) -> index
    // CHECK: handshake.buffer
    %7 = "handshake.branch"(%4) {control = false} : (index) -> index
    // CHECK: handshake.buffer
    %8 = "handshake.mux"(%11#1, %22, %7) : (index, index, index) -> index
    // CHECK: handshake.buffer
    %9:2 = "handshake.fork"(%8) {control = false} : (index) -> (index, index)
    // CHECK: handshake.buffer
    %10:2 = "handshake.control_merge"(%23, %5) {control = true} : (none, none) -> (none, index)
    // CHECK: handshake.buffer
    %11:2 = "handshake.fork"(%10#1) {control = false} : (index) -> (index, index)
    // CHECK: handshake.buffer
    %12 = "handshake.mux"(%11#0, %24, %6) : (index, index, index) -> index
    // CHECK: handshake.buffer
    %13:2 = "handshake.fork"(%12) {control = false} : (index) -> (index, index)
    // CHECK: handshake.buffer
    %14 = cmpi slt, %13#1, %9#1 : index
    // CHECK: handshake.buffer
    %15:3 = "handshake.fork"(%14) {control = false} : (i1) -> (i1, i1, i1)
    // CHECK: handshake.buffer
    %trueResult, %falseResult = "handshake.conditional_branch"(%15#2, %9#0) {control = false} : (i1, index) -> (index, index)
    // CHECK: handshake.buffer
    "handshake.sink"(%falseResult) : (index) -> ()
    // CHECK: handshake.buffer
    %trueResult_0, %falseResult_1 = "handshake.conditional_branch"(%15#1, %10#0) {control = true} : (i1, none) -> (none, none)
    // CHECK: handshake.buffer
    %trueResult_2, %falseResult_3 = "handshake.conditional_branch"(%15#0, %13#0) {control = false} : (i1, index) -> (index, index)
    // CHECK: handshake.buffer
    "handshake.sink"(%falseResult_3) : (index) -> ()
    // CHECK: handshake.buffer
    %16 = "handshake.merge"(%trueResult_2) : (index) -> index
    // CHECK: handshake.buffer
    %17 = "handshake.merge"(%trueResult) : (index) -> index
    // CHECK: handshake.buffer
    %18:2 = "handshake.control_merge"(%trueResult_0) {control = true} : (none) -> (none, index)
    // CHECK: handshake.buffer
    %19:2 = "handshake.fork"(%18#0) {control = true} : (none) -> (none, none)
    // CHECK: handshake.buffer
    "handshake.sink"(%18#1) : (index) -> ()
    // CHECK: handshake.buffer
    %20 = "handshake.constant"(%19#0) {value = 1 : index} : (none) -> index
    // CHECK: handshake.buffer
    %21 = addi %16, %20 : index
    // CHECK: handshake.buffer
    %22 = "handshake.branch"(%17) {control = false} : (index) -> index
    // CHECK: handshake.buffer
    %23 = "handshake.branch"(%19#1) {control = true} : (none) -> none
    // CHECK: handshake.buffer
    %24 = "handshake.branch"(%21) {control = false} : (index) -> index
    // CHECK: handshake.buffer
    %25:2 = "handshake.control_merge"(%falseResult_1) {control = true} : (none) -> (none, index)
    // CHECK: handshake.buffer
    "handshake.sink"(%25#1) : (index) -> ()
    // CHECK: handshake.buffer
    handshake.return %25#0 : none
  }
}
