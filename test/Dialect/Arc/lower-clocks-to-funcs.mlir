// RUN: circt-opt %s --arc-lower-clocks-to-funcs --verify-diagnostics | FileCheck %s

// CHECK-LABEL: func.func @Trivial_clock(%arg0: !arc.storage<42>) {
// CHECK-NEXT:    [[TMP:%.+]] = hw.constant 9001
// CHECK-NEXT:    call @DummyB([[TMP]]) {a}
// CHECK-NEXT:    return
// CHECK-NEXT:  }

// CHECK-LABEL: func.func @Trivial_passthrough(%arg0: !arc.storage<42>) {
// CHECK-NEXT:    [[TMP:%.+]] = hw.constant 9002
// CHECK-NEXT:    call @DummyB([[TMP]]) {b}
// CHECK-NEXT:    return
// CHECK-NEXT:  }

// CHECK-LABEL: func.func @Trivial_initial(%arg0: !arc.storage<42>) {
// CHECK-NEXT:    [[TMP:%.+]] = hw.constant 9003
// CHECK-NEXT:    call @DummyB([[TMP]]) {c}
// CHECK-NEXT:    call @Trivial_passthrough(%arg0)
// CHECK-NEXT:    return
// CHECK-NEXT:  }

// CHECK-LABEL: func.func @Trivial_final(%arg0: !arc.storage<42>) {
// CHECK-NEXT:    [[TMP:%.+]] = hw.constant 9004
// CHECK-NEXT:    call @DummyB([[TMP]]) {d}
// CHECK-NEXT:    return
// CHECK-NEXT:  }

// CHECK-LABEL: arc.model @Trivial
// CHECK-SAME:    io !hw.modty<>
// CHECK-SAME:    initializer @Trivial_initial
// CHECK-SAME:    finalizer @Trivial_final
// CHECK-SAME:  {
// CHECK-NEXT:  ^bb0(%arg0: !arc.storage<42>):
// CHECK-NEXT:    %true = hw.constant true
// CHECK-NEXT:    scf.if %true {
// CHECK-NEXT:      func.call @Trivial_clock(%arg0) : (!arc.storage<42>) -> ()
// CHECK-NEXT:    }
// CHECK-NEXT:    func.call @Trivial_passthrough(%arg0) : (!arc.storage<42>) -> ()
// CHECK-NEXT:  }

arc.model @Trivial io !hw.modty<> {
^bb0(%arg0: !arc.storage<42>):
  %true = hw.constant true
  %0 = hw.constant 9001 : i42
  %1 = hw.constant 9002 : i42
  %2 = hw.constant 9003 : i42
  %3 = hw.constant 9004 : i42
  arc.clock_tree %true {
    func.call @DummyB(%0) {a} : (i42) -> ()
  }
  arc.passthrough {
    func.call @DummyB(%1) {b} : (i42) -> ()
  }
  arc.initial {
    func.call @DummyB(%2) {c} : (i42) -> ()
  }
  arc.final {
    func.call @DummyB(%3) {d} : (i42) -> ()
  }
}

func.func private @DummyA() -> i42
func.func private @DummyB(i42) -> ()

//===----------------------------------------------------------------------===//

// CHECK-LABEL: func.func @NestedRegions_passthrough(%arg0: !arc.storage<42>) {
// CHECK-NEXT:    %true = hw.constant true
// CHECK-NEXT:    scf.if %true {
// CHECK-NEXT:      hw.constant 1337
// CHECK-NEXT:    }
// CHECK-NEXT:    return
// CHECK-NEXT:  }

// CHECK-LABEL: arc.model @NestedRegions io !hw.modty<> {
// CHECK-NEXT:  ^bb0(%arg0: !arc.storage<42>):
// CHECK-NEXT:    func.call @NestedRegions_passthrough(%arg0) : (!arc.storage<42>) -> ()
// CHECK-NEXT:  }

arc.model @NestedRegions io !hw.modty<> {
^bb0(%arg0: !arc.storage<42>):
  arc.passthrough {
    %true = hw.constant true
    scf.if %true {
      %0 = hw.constant 1337 : i42
    }
  }
}

//===----------------------------------------------------------------------===//

// The constants should copied to the top of the clock function body, not in
// front of individual users, to prevent issues with caching and nested regions.
// https://github.com/llvm/circt/pull/4685#discussion_r1132913165

// CHECK-LABEL: func.func @InsertionOrderProblem_passthrough(%arg0: !arc.storage<42>) {
// CHECK-NEXT:    %true = hw.constant true
// CHECK-NEXT:    %false = hw.constant false
// CHECK-NEXT:    scf.if %true {
// CHECK-NEXT:      comb.add %true, %false
// CHECK-NEXT:    }
// CHECK-NEXT:    return
// CHECK-NEXT:  }

// CHECK-LABEL: arc.model @InsertionOrderProblem
arc.model @InsertionOrderProblem io !hw.modty<> {
^bb0(%arg0: !arc.storage<42>):
  %true = hw.constant true
  %false = hw.constant false
  arc.passthrough {
    scf.if %true {
      comb.add %true, %false : i1
    }
  }
}
