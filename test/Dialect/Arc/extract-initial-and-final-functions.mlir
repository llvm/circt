// RUN: circt-opt %s --arc-extract-initial-and-final-functions --verify-diagnostics | FileCheck %s

// CHECK-LABEL: func.func @Trivial_initial(%arg0: !arc.storage) {
// CHECK-NEXT:    [[TMP:%.+]] = hw.constant 9002
// CHECK-NEXT:    call @DummyB([[TMP]]) {b}
// CHECK-NEXT:    return
// CHECK-NEXT:  }

// CHECK-LABEL: func.func @Trivial_final(%arg0: !arc.storage) {
// CHECK-NEXT:    [[TMP:%.+]] = hw.constant 9003
// CHECK-NEXT:    call @DummyB([[TMP]]) {c}
// CHECK-NEXT:    return
// CHECK-NEXT:  }

// CHECK-LABEL: arc.model @Trivial
// CHECK-SAME:    io !hw.modty<>
// CHECK-SAME:    initializer @Trivial_initial
// CHECK-SAME:    finalizer @Trivial_final
// CHECK-SAME:  {
// CHECK-NEXT:  ^bb0(%arg0: !arc.storage):
// CHECK-NEXT:    [[TMP:%.+]] = hw.constant 9001
// CHECK:         call @DummyB([[TMP]]) {a}
// CHECK-NEXT:  }

arc.model @Trivial io !hw.modty<> {
^bb0(%arg0: !arc.storage):
  %0 = hw.constant 9001 : i42
  %1 = hw.constant 9002 : i42
  %2 = hw.constant 9003 : i42
  func.call @DummyB(%0) {a} : (i42) -> ()
  arc.initial {
    func.call @DummyB(%1) {b} : (i42) -> ()
  }
  arc.final {
    func.call @DummyB(%2) {c} : (i42) -> ()
  }
}

func.func private @DummyA() -> i42
func.func private @DummyB(i42) -> ()

// Pure operations are cloned along with their operands; other values must be
// model arguments.
// CHECK-LABEL: func.func @PureChain_initial(%arg0: !arc.storage) {
// CHECK-NEXT:    [[A:%.+]] = hw.constant 1
// CHECK-NEXT:    [[B:%.+]] = comb.add [[A]], [[A]]
// CHECK-NEXT:    [[C:%.+]] = arc.as_context %arg0 : !arc.storage
// CHECK-NEXT:    arc.terminate [[C]], true
// CHECK-NEXT:    call @DummyB([[B]])
// CHECK-NEXT:    return
arc.model @PureChain io !hw.modty<> {
^bb0(%arg0: !arc.storage):
  %0 = hw.constant 1 : i42
  %1 = comb.add %0, %0 : i42
  arc.initial {
    %ctx = arc.as_context %arg0 : !arc.storage
    arc.terminate %ctx, true
    func.call @DummyB(%1) : (i42) -> ()
  }
}
