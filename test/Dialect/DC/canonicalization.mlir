// RUN: circt-opt %s --canonicalize --cse --canonicalize | FileCheck %s
// Waiting on: https://github.com/llvm/llvm-project/issues/64280

// CHECK-LABEL:   func.func @staggeredJoin1(
// CHECK-SAME:                    %[[VAL_0:.*]]: !dc.token,
// CHECK-SAME:                    %[[VAL_1:.*]]: !dc.token) -> !dc.token {
// CHECK:           %[[VAL_2:.*]] = dc.join %[[VAL_0]], %[[VAL_1]]
// CHECK:           return %[[VAL_2]] : !dc.token
// CHECK:         }
func.func @staggeredJoin1(%a: !dc.token, %b : !dc.token) -> (!dc.token) {
    %0 = dc.join %a, %b
    %1 = dc.join %0
    return %1 : !dc.token
}

// CHECK-LABEL:   func.func @staggeredJoin2(
// CHECK-SAME:           %[[VAL_0:.*]]: !dc.token, %[[VAL_1:.*]]: !dc.token, %[[VAL_2:.*]]: !dc.token, %[[VAL_3:.*]]: !dc.token) -> !dc.token {
// CHECK:           %[[VAL_4:.*]] = dc.join %[[VAL_2]], %[[VAL_3]], %[[VAL_0]], %[[VAL_1]]
// CHECK:           return %[[VAL_4]] : !dc.token
// CHECK:         }
func.func @staggeredJoin2(%a: !dc.token, %b : !dc.token, %c : !dc.token, %d : !dc.token) -> (!dc.token) {
    %0 = dc.join %a, %b
    %1 = dc.join %c, %0, %d
    return %1 : !dc.token
}

// CHECK-LABEL:   func.func @redundantJoinOperands(
// CHECK-SAME:                                     %[[VAL_0:.*]]: !dc.token,
// CHECK-SAME:                                     %[[VAL_1:.*]]: !dc.token) -> !dc.token {
// CHECK:           %[[VAL_2:.*]] = dc.join %[[VAL_0]], %[[VAL_1]]
// CHECK:           return %[[VAL_2]] : !dc.token
// CHECK:         }
func.func @redundantJoinOperands(%a: !dc.token, %b : !dc.token) -> (!dc.token) {
    %0 = dc.join %a, %b, %b
    return %0 : !dc.token
}

// CHECK-LABEL:   func.func @fork(
// CHECK-SAME:                    %[[VAL_0:.*]]: !dc.token) -> !dc.token {
// CHECK:           return %[[VAL_0]] : !dc.token
// CHECK:         }
func.func @fork(%a: !dc.token) -> (!dc.token) {
    %0 = dc.fork [1] %a
    return %0 : !dc.token
}

// CHECK-LABEL:   func.func @packUnpack(
// CHECK-SAME:                          %[[VAL_0:.*]]: !dc.token,
// CHECK-SAME:                          %[[VAL_1:.*]]: i32) -> (!dc.token, i32) {
// CHECK:           return %[[VAL_0]], %[[VAL_1]] : !dc.token, i32
// CHECK:         }
func.func @packUnpack(%a: !dc.token, %b : i32) -> (!dc.token, i32) {
    %0 = dc.pack %a, %b : i32
    %1:2 = dc.unpack %0 : !dc.value<i32>
    return %1#0, %1#1 : !dc.token, i32
}

// CHECK-LABEL:   func.func @redundantPack(
// CHECK-SAME:                             %[[VAL_0:.*]]: !dc.token,
// CHECK-SAME:                             %[[VAL_1:.*]]: i32) -> !dc.token {
// CHECK:           return %[[VAL_0]] : !dc.token
// CHECK:         }
func.func @redundantPack(%a: !dc.token, %b : i32) -> (!dc.token) {
    %0 = dc.pack %a, %b : i32 
    %1:2 = dc.unpack %0 : !dc.value<i32>
    return %1#0 : !dc.token
}

// CHECK-LABEL:   func.func @csePack(
// CHECK-SAME:                       %[[VAL_0:.*]]: !dc.token,
// CHECK-SAME:                       %[[VAL_1:.*]]: i32) -> (!dc.value<i32>, !dc.value<i32>) {
// CHECK:           %[[VAL_3:.*]] = dc.pack %[[VAL_0]], %[[VAL_1]] : i32
// CHECK:           return %[[VAL_3]], %[[VAL_3]] : !dc.value<i32>, !dc.value<i32>
// CHECK:         }
func.func @csePack(%a: !dc.token, %b : i32) -> (!dc.value<i32>, !dc.value<i32>) {
    %0 = dc.pack %a, %b : i32
    %1 = dc.pack %a, %b : i32
    return %0, %1 : !dc.value<i32>, !dc.value<i32>
}


// CHECK-LABEL:   func.func @forkToFork(
// CHECK-SAME:                          %[[VAL_0:.*]]: !dc.token) -> (!dc.token, !dc.token, !dc.token) {
// CHECK:           %[[VAL_1:.*]]:3 = dc.fork [3] %[[VAL_0]]
// CHECK:           return %[[VAL_1]]#0, %[[VAL_1]]#1, %[[VAL_1]]#2 : !dc.token, !dc.token, !dc.token
// CHECK:         }
func.func @forkToFork(%a: !dc.token) -> (!dc.token, !dc.token, !dc.token) {
    %0, %1 = dc.fork [2] %a
    %2, %3 = dc.fork [2] %0
    return %1, %2, %3 : !dc.token, !dc.token, !dc.token
}

// CHECK-LABEL:   func.func @forkToFork2(
// CHECK-SAME:                           %[[VAL_0:.*]]: !dc.token) -> (!dc.token, !dc.token, !dc.token) {
// CHECK:           %[[VAL_1:.*]]:3 = dc.fork [3] %[[VAL_0]]
// CHECK:           return %[[VAL_1]]#0, %[[VAL_1]]#1, %[[VAL_1]]#2 : !dc.token, !dc.token, !dc.token
// CHECK:         }
func.func @forkToFork2(%a: !dc.token) -> (!dc.token, !dc.token, !dc.token) {
    %0, %1 = dc.fork [2] %a
    %2, %3 = dc.fork [2] %1
    return %0, %2, %3 : !dc.token, !dc.token, !dc.token
}

// CHECK-LABEL:   func.func @merge(
// CHECK-SAME:                     %[[VAL_0:.*]]: !dc.value<i1>) -> !dc.token {
// CHECK:           %[[VAL_1:.*]], %[[VAL_2:.*]] = dc.unpack %[[VAL_0]] : !dc.value<i1>
// CHECK:           return %[[VAL_1]] : !dc.token
// CHECK:         }
func.func @merge(%sel : !dc.value<i1>) -> (!dc.token) {
    // Canonicalize away a merge that is fed by a branch with the same select
    // input.
    %true, %false = dc.branch %sel
    %0 = dc.select %sel, %true, %false
    return %0 : !dc.token
}

// CHECK-LABEL:   func.func @joinOnSource(
// CHECK-SAME:                            %[[VAL_0:.*]]: !dc.token,
// CHECK-SAME:                            %[[VAL_1:.*]]: !dc.token) -> !dc.token {
// CHECK:           %[[VAL_2:.*]] = dc.join %[[VAL_0]], %[[VAL_1]]
// CHECK:           return %[[VAL_2]] : !dc.token
// CHECK:         }
func.func @joinOnSource(%a : !dc.token, %b : !dc.token) -> (!dc.token) {
    %0 = dc.source
    %out = dc.join %a, %0, %b
    return %out : !dc.token
}


// Join on branch, where all branch results are used in the join is a no-op,
// and the join can use the token of the input value to the branch.
// CHECK-LABEL:   func.func @joinOnBranch(
// CHECK-SAME:                            %[[VAL_0:.*]]: !dc.value<i1>, %[[VAL_1:.*]]: !dc.value<i1>, %[[VAL_2:.*]]: !dc.token) -> (!dc.token, !dc.token) {
// CHECK:           %[[VAL_3:.*]], %[[VAL_4:.*]] = dc.branch %[[VAL_1]]
// CHECK:           %[[VAL_5:.*]], %[[VAL_6:.*]] = dc.unpack %[[VAL_0]] : !dc.value<i1>
// CHECK:           %[[VAL_7:.*]] = dc.join %[[VAL_2]], %[[VAL_3]], %[[VAL_5]]
// CHECK:           return %[[VAL_7]], %[[VAL_4]] : !dc.token, !dc.token
// CHECK:         }
func.func @joinOnBranch(%sel : !dc.value<i1>, %sel2 : !dc.value<i1>, %other : !dc.token) -> (!dc.token, !dc.token) {
    %true, %false = dc.branch %sel
    %true2, %false2 = dc.branch %sel2
    %out = dc.join %true, %false, %other, %true2
    return %out, %false2 : !dc.token, !dc.token
}

// CHECK-LABEL:   func.func @forkOfSource() -> (!dc.token, !dc.token) {
// CHECK:           %[[VAL_0:.*]] = dc.source
// CHECK:           return %[[VAL_0]], %[[VAL_0]] : !dc.token, !dc.token
// CHECK:         }
func.func @forkOfSource() -> (!dc.token, !dc.token) {
    %0 = dc.source
    %1:2 = dc.fork [2] %0
    return %1#0, %1#1 : !dc.token, !dc.token
}

// CHECK-LABEL:  hw.module @TestForkCanonicalization(in %arg0 : !dc.token, out out0 : !dc.token, out out1 : !dc.token) {
// CHECK-NEXT:       [[R0:%.+]]:2 = dc.fork [2] %arg0
// CHECK-NEXT:       [[R1:%.+]] = dc.buffer[4] [[R0]]#0 : !dc.token
// CHECK-NEXT:       [[R2:%.+]] = dc.buffer[4] [[R0]]#1 : !dc.token
// CHECK-NEXT:       hw.output [[R1]], [[R2]] : !dc.token, !dc.token

hw.module @TestForkCanonicalization(in %arg0: !dc.token, out out0: !dc.token, out out1: !dc.token) {
  %0:3 = dc.fork [3] %arg0
  %1 = dc.buffer [4] %0#1 : !dc.token
  %2 = dc.buffer [4] %0#2 : !dc.token
  dc.sink %0#2
  hw.output %1, %2 : !dc.token, !dc.token
}
