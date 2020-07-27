func @simple_loop() {
^bb0:
  br ^bb1
^bb1:	// pred: ^bb0
  %c1 = constant 1 : index
  %c42 = constant 42 : index
  br ^bb2(%c1 : index)
^bb2(%0: index):	// 2 preds: ^bb1, ^bb3
  %1 = cmpi "slt", %0, %c42 : index
  cond_br %1, ^bb3, ^bb4
^bb3:	// pred: ^bb2
  %c1_0 = constant 1 : index
  %2 = addi %0, %c1_0 : index
  br ^bb2(%2 : index)
^bb4:	// pred: ^bb2
  return
}


func @simple_loop() {
^bb0:
  br ^bb1

^bb1:	// pred: ^bb0
  %10:2 = staticlogic.pipeline() {
    %c1 = constant 1: index
    %c42 = constant 42: index
    return %c1, %c42: index, index
  }: () -> (index, index);
  
  br ^bb2(%10#0: index, %10#1: index)

^bb2(%0: index, %11: index):	// 2 preds: ^bb1, ^bb3
  %12:2 = staticlogic.pipeline(%0, %11) {
    %1 = cmpi "slt", %0, %11: index
    return %0, %1: index, index
  }: (index, index) -> (index, index)

  cond_br %12#1, ^bb3(%12#0: index), ^bb4

^bb3(%0: index):	// pred: ^bb2
  %13:2 = staticlogic.pipeline(%0: index) {
    %c1_0 = constant 1: index
    %2 = addi %0, %c1_0: index
    return %0, %2: index, index
  }: index -> (index, index)

  br ^bb2(%13#0, %13#1: index, index)

^bb4:	// pred: ^bb2
  return
}


module {
  handshake.func @simple_loop(%arg0: none, ...) -> none {
    %0 = "handshake.branch"(%arg0) {control = true} : (none) -> none
    %1:2 = "handshake.control_merge"(%0) {control = true} : (none) -> (none, index)
    %2:3 = "handshake.fork"(%1#0) {control = true} : (none) -> (none, none, none)
    "handshake.sink"(%1#1) : (index) -> ()
    %3 = "handshake.constant"(%2#1) {value = 1 : index} : (none) -> index
    %4 = "handshake.constant"(%2#0) {value = 42 : index} : (none) -> index
    %5 = "handshake.branch"(%2#2) {control = true} : (none) -> none
    %6 = "handshake.branch"(%3) {control = false} : (index) -> index
    %7 = "handshake.branch"(%4) {control = false} : (index) -> index
    %8 = "handshake.mux"(%11#1, %22, %7) : (index, index, index) -> index
    %9:2 = "handshake.fork"(%8) {control = false} : (index) -> (index, index)
    %10:2 = "handshake.control_merge"(%23, %5) {control = true} : (none, none) -> (none, index)
    %11:2 = "handshake.fork"(%10#1) {control = false} : (index) -> (index, index)
    %12 = "handshake.mux"(%11#0, %24, %6) : (index, index, index) -> index
    %13:2 = "handshake.fork"(%12) {control = false} : (index) -> (index, index)
    %14 = cmpi "slt", %13#1, %9#1 : index
    %15:3 = "handshake.fork"(%14) {control = false} : (i1) -> (i1, i1, i1)
    %trueResult, %falseResult = "handshake.conditional_branch"(%15#2, %9#0) {control = false} : (i1, index) -> (index, index)
    "handshake.sink"(%falseResult) : (index) -> ()
    %trueResult_0, %falseResult_1 = "handshake.conditional_branch"(%15#1, %10#0) {control = true} : (i1, none) -> (none, none)
    %trueResult_2, %falseResult_3 = "handshake.conditional_branch"(%15#0, %13#0) {control = false} : (i1, index) -> (index, index)
    "handshake.sink"(%falseResult_3) : (index) -> ()
    %16 = "handshake.merge"(%trueResult_2) : (index) -> index
    %17 = "handshake.merge"(%trueResult) : (index) -> index
    %18:2 = "handshake.control_merge"(%trueResult_0) {control = true} : (none) -> (none, index)
    %19:2 = "handshake.fork"(%18#0) {control = true} : (none) -> (none, none)
    "handshake.sink"(%18#1) : (index) -> ()
    %20 = "handshake.constant"(%19#0) {value = 1 : index} : (none) -> index
    %21 = addi %16, %20 : index
    %22 = "handshake.branch"(%17) {control = false} : (index) -> index
    %23 = "handshake.branch"(%19#1) {control = true} : (none) -> none
    %24 = "handshake.branch"(%21) {control = false} : (index) -> index
    %25:2 = "handshake.control_merge"(%falseResult_1) {control = true} : (none) -> (none, index)
    "handshake.sink"(%25#1) : (index) -> ()
    handshake.return %25#0 : none
  }
}