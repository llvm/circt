// RUN: circt-opt %s --arc-group-computation | FileCheck %s

func.func private @F0() -> i1
func.func private @F1(i1) -> (i1)
func.func private @F2(i1, i1) -> (i1)
func.func private @F3() -> (i1, i1, i1)
func.func private @F4(i1, i1, i1, i1) -> (i1)
func.func private @F5(i1, i1) -> ()
func.func private @F6(i1)
func.func private @F7(i1) -> (i1, i1)

// CHECK-LABEL: @Case1
func.func @Case1(%arg0: i1) {
  arc.clock_tree %arg0 {
    %0 = func.call @F0() : () -> i1
    %1 = func.call @F1(%0) : (i1) -> i1
    %2 = func.call @F2(%0, %1) : (i1, i1) -> i1

    %3 = func.call @F0() : () -> i1
    %4 = func.call @F1(%3) : (i1) -> i1
    %5 = func.call @F2(%3, %4) : (i1, i1) -> i1

    %6 = func.call @F2(%2, %5) : (i1, i1) -> i1

    %7 = func.call @F1(%6) : (i1) -> i1
    %8 = func.call @F1(%7) : (i1) -> i1
    %9 = func.call @F2(%7, %8) : (i1, i1) -> i1
  }
  return
}

// CHECK-LABEL: @Case2
func.func @Case2(%arg0: i1) {
  arc.clock_tree %arg0 {
    %0:3 = func.call @F3() : () -> (i1, i1, i1)
    %1:3 = func.call @F3() : () -> (i1, i1, i1)
    %2 = func.call @F4(%0#0, %0#1, %0#2, %1#0) : (i1, i1, i1, i1) -> i1
    %3 = func.call @F4(%1#0, %1#1, %1#2, %0#0) : (i1, i1, i1, i1) -> i1
    func.call @F5(%2, %3) : (i1, i1) -> ()
  }
  return
}

// CHECK-LABEL: @Case3
func.func @Case3(%arg0: i1) {
  arc.clock_tree %arg0 {
    %0 = func.call @F0() : () -> i1
    func.call @F6(%0) : (i1) -> ()
    %1 = func.call @F0() : () -> i1
    %2 = func.call @F1(%1) : (i1) -> i1
    %3 = func.call @F0() : () -> i1
    %4 = func.call @F0() : () -> i1
    %5 = func.call @F4(%1, %2, %3, %4) : (i1, i1, i1, i1) -> i1
    func.call @F6(%5) : (i1) -> ()
    %9 = func.call @F2(%3, %4) : (i1, i1) -> i1
    func.call @F6(%9) : (i1) -> ()
  }
  return
}

// CHECK-LABEL: @Case4
func.func @Case4(%arg0: i1) {
  arc.clock_tree %arg0 {
    %0 = func.call @F0() : () -> i1
    %1 = func.call @F0() : () -> i1
    %2 = func.call @F2(%1, %0) : (i1, i1) -> i1
    %3 = func.call @F2(%0, %1) : (i1, i1) -> i1
    %4:2 = func.call @F7(%3) : (i1) -> (i1, i1)
    %5 = func.call @F1(%4#0) : (i1) -> i1
    %6 = func.call @F2(%2, %5) : (i1, i1) -> i1
    %7 = func.call @F1(%6) : (i1) -> i1
    %8 = func.call @F1(%7) : (i1) -> i1
    %9 = func.call @F1(%8) : (i1) -> i1
    func.call @F6(%9) : (i1) -> ()
  }
  return
}

// CHECK-LABEL: @Case5
func.func @Case5(%arg0: i1) {
  arc.clock_tree %arg0 {
    %0 = func.call @F0() : () -> i1
    %1 = func.call @F1(%0) : (i1) -> i1
    %2 = func.call @F1(%1) : (i1) -> i1
    %3 = func.call @F0() : () -> i1
    %4 = func.call @F2(%3, %2) : (i1, i1) -> i1
    %5 = func.call @F1(%4) : (i1) -> i1
    func.call @F6(%5) : (i1) -> ()
    %6 = func.call @F2(%3, %2) : (i1, i1) -> i1
    func.call @F6(%6) : (i1) -> ()
  }
  return
}
