// RUN: circt-opt %s --arc-group-resets-and-enables | FileCheck %s

// CHECK-LABEL: arc.model "BasicResetGrouping"
arc.model "BasicResetGrouping" {
^bb0(%arg0: !arc.storage):
  %c0_i4 = hw.constant 0 : i4
  // CHECK: [[IN_CLOCK_STATE:%.+]] = arc.alloc
  %in_clock = arc.alloc %arg0 : (!arc.storage) -> !arc.state<i1>
  // CHECK: [[IN_I0_STATE:%.+]] = arc.alloc
  %in_i0 = arc.alloc %arg0 : (!arc.storage) -> !arc.state<i4>
  // CHECK: [[IN_I1_STATE:%.+]] = arc.alloc
  %in_i1 = arc.alloc %arg0 : (!arc.storage) -> !arc.state<i4>
  // CHECK: [[IN_RESET0_STATE:%.+]] = arc.alloc
  %in_reset0 = arc.alloc %arg0 : (!arc.storage) -> !arc.state<i1>
  // CHECK: [[IN_RESET1_STATE:%.+]] = arc.alloc
  %in_reset1 = arc.alloc %arg0 : (!arc.storage) -> !arc.state<i1>
  %0 = arc.state_read %in_clock : <i1>
  // Group resets:
  arc.clock_tree %0 {
    //  CHECK: [[IN_RESET0:%.+]] = arc.state_read [[IN_RESET0_STATE]]
    %3 = arc.state_read %in_reset0 : <i1>
    //  CHECK-NEXT: scf.if [[IN_RESET0]] {
    scf.if %3 {
      //   CHECK-NEXT:  arc.state_write [[FOO_ALLOC:%.+]] = %c0_i4
      //   CHECK-NEXT:  arc.state_write [[BAR_ALLOC:%.+]] = %c0_i4
      arc.state_write %1 = %c0_i4 : <i4>
      //   CHECK-NEXT: } else {
    } else {
      // CHECK-NEXT:  [[IN_I0:%.+]] = arc.state_read [[IN_I0_STATE]]
      // CHECK-NEXT:  arc.state_write [[FOO_ALLOC]] = [[IN_I0]]
      // CHECK-NEXT:  [[IN_I1:%.+]] = arc.state_read [[IN_I1_STATE]]
      // CHECK-NEXT:  arc.state_write [[BAR_ALLOC]] = [[IN_I1]]
      %4 = arc.state_read %in_i0 : <i4>
      arc.state_write %1 = %4 : <i4>
      // CHECK-NEXT: }
    }
    scf.if %3 {
      arc.state_write %2 = %c0_i4 : <i4>
    } else {
      %5 = arc.state_read %in_i1 : <i4>
      arc.state_write %2 = %5 : <i4>
    }
    // CHECK-NEXT: }
  }
  // Don't group resets that don't match:
  arc.clock_tree %0 {
    //  CHECK: [[IN_RESET0_1:%.+]] = arc.state_read [[IN_RESET0_STATE]]
    %6 = arc.state_read %in_reset0 : <i1>
    //  CHECK-NEXT: [[IN_RESET1_1:%.+]] = arc.state_read [[IN_RESET1_STATE]]
    %7 = arc.state_read %in_reset1 : <i1>
    //  CHECK-NEXT: scf.if [[IN_RESET0_1]] {
    scf.if %6 {
      //   CHECK-NEXT:  arc.state_write [[FOO_ALLOC]] = %c0_i4
      arc.state_write %1 = %c0_i4 : <i4>
      //   CHECK-NEXT: } else {
    } else {
      // CHECK-NEXT:  [[IN_I0_1:%.+]] = arc.state_read [[IN_I0_STATE]]
      // CHECK-NEXT:  arc.state_write [[FOO_ALLOC]] = [[IN_I0_1]]
      %8 = arc.state_read %in_i0 : <i4>
      arc.state_write %1 = %8 : <i4>
      // CHECK-NEXT: }
    }
    //  CHECK-NEXT: scf.if [[IN_RESET1_1]] {
    scf.if %7 {
      //   CHECK-NEXT:  arc.state_write [[BAR_ALLOC]] = %c0_i4
      arc.state_write %2 = %c0_i4 : <i4>
      //   CHECK-NEXT: } else {
    } else {
      // CHECK-NEXT:  [[IN_I1_1:%.+]] = arc.state_read [[IN_I1_STATE]]
      // CHECK-NEXT:  arc.state_write [[BAR_ALLOC]] = [[IN_I1_1]]
      %9 = arc.state_read %in_i1 : <i4>
      arc.state_write %2 = %9 : <i4>
    }
    // CHECK-NEXT: }
  // CHECK-NEXT: }
  }
  // Don't group IfOps with return values:
  arc.clock_tree %0 {
    //  CHECK: [[IN_RESET0:%.+]] = arc.state_read [[IN_RESET0_STATE]]
    %10 = arc.state_read %in_reset0 : <i1>
    //  CHECK-NEXT: scf.if [[IN_RESET0]] {
    scf.if %10 {
      //   CHECK-NEXT:  arc.state_write [[FOO_ALLOC]] = %c0_i4
      arc.state_write %1 = %c0_i4 : <i4>
      //   CHECK-NEXT: } else {
    } else {
      // CHECK-NEXT:  [[IN_I0:%.+]] = arc.state_read [[IN_I0_STATE]]
      // CHECK-NEXT:  arc.state_write [[FOO_ALLOC]] = [[IN_I0]]
      %11 = arc.state_read %in_i0 : <i4>
      arc.state_write %1 = %11 : <i4>
      // CHECK-NEXT: }
    }
    //  CHECK-NEXT: [[IF_RESULT:%.+]] scf.if [[IN_RESET0]] -> (i4) {
    %res = scf.if %10 -> (i4) {
      //   CHECK-NEXT:  arc.state_write [[BAR_ALLOC]] = %c0_i4
      //   CHECK-NEXT:  scf.yield %c0_i4 : i4
      arc.state_write %2 = %c0_i4 : <i4>
      scf.yield %c0_i4 : i4
      //   CHECK-NEXT: } else {
    } else {
      // CHECK-NEXT:  [[IN_I1:%.+]] = arc.state_read [[IN_I1_STATE]]
      // CHECK-NEXT:  arc.state_write [[BAR_ALLOC]] = [[IN_I1]]
      //   CHECK-NEXT:  scf.yield %c0_i4 : i4
      %12 = arc.state_read %in_i1 : <i4>
      arc.state_write %2 = %12 : <i4>
      scf.yield %c0_i4 : i4
    }
    // CHECK-NEXT: }
  // CHECK-NEXT: }
  }
  // Group resets with no else in an early block (that has its contents moved):
  arc.clock_tree %0 {
    //  CHECK: [[IN_RESET0:%.+]] = arc.state_read [[IN_RESET0_STATE]]
    %13 = arc.state_read %in_reset0 : <i1>
    //  CHECK-NEXT: scf.if [[IN_RESET0]] {
    scf.if %13 {
      //   CHECK-NEXT:  arc.state_write [[FOO_ALLOC:%.+]] = %c0_i4
      //   CHECK-NEXT:  arc.state_write [[BAR_ALLOC:%.+]] = %c0_i4
      arc.state_write %1 = %c0_i4 : <i4>
      //   CHECK-NEXT: } else {
      // CHECK-NEXT:  [[IN_I1:%.+]] = arc.state_read [[IN_I1_STATE]]
      // CHECK-NEXT:  arc.state_write [[BAR_ALLOC]] = [[IN_I1]]
      // CHECK-NEXT: }
    }
    scf.if %13 {
      arc.state_write %2 = %c0_i4 : <i4>
    } else {
      %14 = arc.state_read %in_i1 : <i4>
      arc.state_write %2 = %14 : <i4>
    }
    // CHECK-NEXT: }
  }
  // Group resets with no else in the last if (where contents are moved to):
  arc.clock_tree %0 {
    //  CHECK: [[IN_RESET0:%.+]] = arc.state_read [[IN_RESET0_STATE]]
    %15 = arc.state_read %in_reset0 : <i1>
    //  CHECK-NEXT: scf.if [[IN_RESET0]] {
    scf.if %15 {
      //   CHECK-NEXT:  arc.state_write [[FOO_ALLOC:%.+]] = %c0_i4
      //   CHECK-NEXT:  arc.state_write [[BAR_ALLOC:%.+]] = %c0_i4
      arc.state_write %1 = %c0_i4 : <i4>
      //   CHECK-NEXT: } else {
    } else {
      // CHECK-NEXT:  [[IN_I0:%.+]] = arc.state_read [[IN_I0_STATE]]
      // CHECK-NEXT:  arc.state_write [[FOO_ALLOC]] = [[IN_I0]]
      %16 = arc.state_read %in_i0 : <i4>
      arc.state_write %1 = %16 : <i4>
      // CHECK-NEXT: }
    }
    scf.if %15 {
      arc.state_write %2 = %c0_i4 : <i4>
    }
    // CHECK-NEXT: }
  }
  // CHECK-NEXT: [[FOO_ALLOC]] = arc.alloc
  // CHECK-NEXT: [[BAR_ALLOC]] = arc.alloc
  %1 = arc.alloc %arg0 : (!arc.storage) -> !arc.state<i4>
  %2 = arc.alloc %arg0 : (!arc.storage) -> !arc.state<i4>
}

// CHECK-LABEL: arc.model "BasicEnableGrouping"
arc.model "BasicEnableGrouping" {
^bb0(%arg0: !arc.storage):
  %c0_i4 = hw.constant 0 : i4
  // CHECK: [[IN_CLOCK_STATE:%.+]] = arc.alloc
  %in_clock = arc.alloc %arg0 : (!arc.storage) -> !arc.state<i1>
  // CHECK: [[IN_I0_STATE:%.+]] = arc.alloc
  %in_i0 = arc.alloc %arg0 : (!arc.storage) -> !arc.state<i4>
  // CHECK: [[IN_I1_STATE:%.+]] = arc.alloc
  %in_i1 = arc.alloc %arg0 : (!arc.storage) -> !arc.state<i4>
  // CHECK: [[IN_EN0_STATE:%.+]] = arc.alloc
  %in_en0 = arc.alloc %arg0 : (!arc.storage) -> !arc.state<i1>
  // CHECK: [[IN_EN1_STATE:%.+]] = arc.alloc
  %in_en1 = arc.alloc %arg0 : (!arc.storage) -> !arc.state<i1>
  %0 = arc.state_read %in_clock : <i1>
  // Group enables:
  arc.clock_tree %0 {
    //  CHECK: [[IN_EN0:%.+]] = arc.state_read [[IN_EN0_STATE]]
    %3 = arc.state_read %in_en0 : <i1>
    //   CHECK-NEXT: arc.state_write [[FOO_ALLOC:%.+]] = %c0_i4
    //   CHECK-NEXT: arc.state_write [[BAR_ALLOC:%.+]] = %c0_i4
    arc.state_write %1 = %c0_i4 : <i4>
    arc.state_write %2 = %c0_i4 : <i4>
    // CHECK-NEXT:   [[IN_I0:%.+]] = arc.state_read [[IN_I0_STATE]]
    // CHECK-NEXT:   [[IN_I1:%.+]] = arc.state_read [[IN_I1_STATE]]
    // CHECK-NEXT:   scf.if [[IN_EN0]] {
    // CHECK-NEXT:    arc.state_write [[FOO_ALLOC]] = [[IN_I0]]
    // CHECK-NEXT:    arc.state_write [[BAR_ALLOC]] = [[IN_I1]]
    %4 = arc.state_read %in_i0 : <i4>
    arc.state_write %1 = %4 if %3 : <i4>
    %5 = arc.state_read %in_i1 : <i4>
    arc.state_write %2 = %5 if %3 : <i4>
    // CHECK-NEXT:  }
  // CHECK-NEXT: }
  }
  // Don't group non-matching enables:
  arc.clock_tree %0 {
    //  CHECK: [[IN_EN0_1:%.+]] = arc.state_read [[IN_EN0_STATE]]
    %6 = arc.state_read %in_en0 : <i1>
    //  CHECK-NEXT: [[IN_EN1_1:%.+]] = arc.state_read [[IN_EN1_STATE]]
    %7 = arc.state_read %in_en1 : <i1>
    //   CHECK-NEXT: arc.state_write [[FOO_ALLOC]] = %c0_i4
    //   CHECK-NEXT: arc.state_write [[BAR_ALLOC]] = %c0_i4
    arc.state_write %1 = %c0_i4 : <i4>
    arc.state_write %2 = %c0_i4 : <i4>
    // CHECK-NEXT:   [[IN_I0_1:%.+]] = arc.state_read [[IN_I0_STATE]]
    // CHECK-NEXT:   arc.state_write [[FOO_ALLOC]] = [[IN_I0_1]] if [[IN_EN0_1]]
    // CHECK-NEXT:   [[IN_I1_1:%.+]] = arc.state_read [[IN_I1_STATE]]
    // CHECK-NEXT:   arc.state_write [[BAR_ALLOC]] = [[IN_I1_1]] if [[IN_EN1_1]]
    %8 = arc.state_read %in_i0 : <i4>
    arc.state_write %1 = %8 if %6 : <i4>
    %9 = arc.state_read %in_i1 : <i4>
    arc.state_write %2 = %9 if %7 : <i4>
  // CHECK-NEXT: }
  }
  // CHECK-NEXT: [[FOO_ALLOC]] = arc.alloc
  // CHECK-NEXT: [[BAR_ALLOC]] = arc.alloc
  %1 = arc.alloc %arg0 : (!arc.storage) -> !arc.state<i4>
  %2 = arc.alloc %arg0 : (!arc.storage) -> !arc.state<i4>
}

// CHECK-LABEL: arc.model "ResetAndEnableGrouping"
arc.model "ResetAndEnableGrouping" {
^bb0(%arg0: !arc.storage):
  %c0_i4 = hw.constant 0 : i4
  // CHECK: [[IN_CLOCK_STATE:%.+]] = arc.alloc
  %in_clock = arc.alloc %arg0 : (!arc.storage) -> !arc.state<i1>
  // CHECK: [[IN_I0_STATE:%.+]] = arc.alloc
  %in_i0 = arc.alloc %arg0 : (!arc.storage) -> !arc.state<i4>
  // CHECK: [[IN_I1_STATE:%.+]] = arc.alloc
  %in_i1 = arc.alloc %arg0 : (!arc.storage) -> !arc.state<i4>
  // CHECK: [[IN_RESET_STATE:%.+]] = arc.alloc
  %in_reset = arc.alloc %arg0 : (!arc.storage) -> !arc.state<i1>
  // CHECK: [[IN_EN0_STATE:%.+]] = arc.alloc
  %in_en0 = arc.alloc %arg0 : (!arc.storage) -> !arc.state<i1>
  // CHECK: [[IN_EN1_STATE:%.+]] = arc.alloc
  %in_en1 = arc.alloc %arg0 : (!arc.storage) -> !arc.state<i1>
  %0 = arc.state_read %in_clock : <i1>
  // Group enables inside resets:
  arc.clock_tree %0 {
    //  CHECK: [[IN_EN:%.+]] = arc.state_read [[IN_EN1_STATE]]
    %3 = arc.state_read %in_en1 : <i1>
    //  CHECK-NEXT: [[IN_RESET:%.+]] = arc.state_read [[IN_RESET_STATE]]
    %4 = arc.state_read %in_reset : <i1>
    //  CHECK-NEXT: scf.if [[IN_RESET]] {
    scf.if %4 {
      //   CHECK-NEXT: arc.state_write [[FOO_ALLOC:%.+]] = %c0_i4
      //   CHECK-NEXT: arc.state_write [[BAR_ALLOC:%.+]] = %c0_i4
      arc.state_write %1 = %c0_i4 : <i4>
      arc.state_write %2 = %c0_i4 : <i4>
      //   CHECK-NEXT: } else {
    } else {
      // CHECK-NEXT:   [[IN_I0:%.+]] = arc.state_read [[IN_I0_STATE]]
      // CHECK-NEXT:   [[IN_I1:%.+]] = arc.state_read [[IN_I1_STATE]]
      // CHECK-NEXT:   scf.if [[IN_EN]] {
      // CHECK-NEXT:    arc.state_write [[FOO_ALLOC]] = [[IN_I0]]
      // CHECK-NEXT:    arc.state_write [[BAR_ALLOC]] = [[IN_I1]]
      %5 = arc.state_read %in_i0 : <i4>
      arc.state_write %1 = %5 if %3 : <i4>
      %6 = arc.state_read %in_i1 : <i4>
      arc.state_write %2 = %6 if %3 : <i4>
      // CHECK-NEXT:   }
    // CHECK-NEXT:  }
    }
  // CHECK-NEXT: }
  }
  // Group both resets and enables:
  arc.clock_tree %0 {
    //  CHECK: [[IN_RESET:%.+]] = arc.state_read [[IN_RESET_STATE]]
    %7 = arc.state_read %in_reset : <i1>
    //  CHECK-NEXT: [[IN_EN0:%.+]] = arc.state_read [[IN_EN0_STATE]]
    %8 = arc.state_read %in_en0 : <i1>
    //  CHECK-NEXT: scf.if [[IN_RESET]] {
    scf.if %7 {
      //   CHECK-NEXT:  arc.state_write [[FOO_ALLOC]] = %c0_i4
      //   CHECK-NEXT:  arc.state_write [[BAR_ALLOC]] = %c0_i4
      arc.state_write %1 = %c0_i4 : <i4>
      //   CHECK-NEXT: } else {
    } else {
      // CHECK-NEXT:   [[IN_I0:%.+]] = arc.state_read [[IN_I0_STATE]]
      // CHECK-NEXT:   [[IN_I1:%.+]] = arc.state_read [[IN_I1_STATE]]
      // CHECK-NEXT:   scf.if [[IN_EN0]] {
      // CHECK-NEXT:    arc.state_write [[FOO_ALLOC]] = [[IN_I0]]
      // CHECK-NEXT:    arc.state_write [[BAR_ALLOC]] = [[IN_I1]]
      // CHECK-NEXT:   }
      %9 = arc.state_read %in_i0 : <i4>
      arc.state_write %1 = %9 if %8 : <i4>
      // CHECK-NEXT: }
    }
    scf.if %7 {
      arc.state_write %2 = %c0_i4 : <i4>
    } else {
      %10 = arc.state_read %in_i1 : <i4>
      arc.state_write %2 = %10 if %8 : <i4>
    }
    // CHECK-NEXT: }
  }
  // Group resets that are separated by an enable read:
  arc.clock_tree %0 {
    //  CHECK: [[IN_RESET:%.+]] = arc.state_read [[IN_RESET_STATE]]
    %11 = arc.state_read %in_reset : <i1>
    //  CHECK-NEXT: [[IN_EN0:%.+]] = arc.state_read [[IN_EN0_STATE]]
    %12 = arc.state_read %in_en0 : <i1>
    //  CHECK-NEXT: [[IN_EN1:%.+]] = arc.state_read [[IN_EN1_STATE]]
    //  CHECK-NEXT: scf.if [[IN_RESET]] {
    scf.if %11 {
      //   CHECK-NEXT:  arc.state_write [[FOO_ALLOC]] = %c0_i4
      //   CHECK-NEXT:  arc.state_write [[BAR_ALLOC]] = %c0_i4
      arc.state_write %1 = %c0_i4 : <i4>
      //   CHECK-NEXT: } else {
    } else {
      // CHECK-NEXT:  [[IN_I0:%.+]] = arc.state_read [[IN_I0_STATE]]
      // CHECK-NEXT:  arc.state_write [[FOO_ALLOC]] = [[IN_I0]] if [[IN_EN0]]
      // CHECK-NEXT:  [[IN_I1:%.+]] = arc.state_read [[IN_I1_STATE]]
      // CHECK-NEXT:  arc.state_write [[BAR_ALLOC]] = [[IN_I1]] if [[IN_EN1]]
      %13 = arc.state_read %in_i0 : <i4>
      arc.state_write %1 = %13 if %12 : <i4>
      // CHECK-NEXT: }
    }
    %14 = arc.state_read %in_en1 : <i1>
    scf.if %11 {
      arc.state_write %2 = %c0_i4 : <i4>
    } else {
      %15 = arc.state_read %in_i1 : <i4>
      arc.state_write %2 = %15 if %14 : <i4>
    }
    // CHECK-NEXT: }
  }
  // CHECK-NEXT: [[FOO_ALLOC]] = arc.alloc
  // CHECK-NEXT: [[BAR_ALLOC]] = arc.alloc
  %1 = arc.alloc %arg0 : (!arc.storage) -> !arc.state<i4>
  %2 = arc.alloc %arg0 : (!arc.storage) -> !arc.state<i4>
}
