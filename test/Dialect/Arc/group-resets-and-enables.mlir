// RUN: circt-opt %s --arc-group-resets-and-enables | FileCheck %s

// CHECK-LABEL: arc.model "JustGroupResets"
arc.model "JustGroupResets" {
^bb0(%arg0: !arc.storage):
  %c0_i4 = hw.constant 0 : i4
  %true = hw.constant true
  %in_clock = arc.root_input "clock", %arg0 : (!arc.storage) -> !arc.state<i1>
  %in_i0 = arc.root_input "i0", %arg0 : (!arc.storage) -> !arc.state<i4>
  %in_i1 = arc.root_input "i1", %arg0 : (!arc.storage) -> !arc.state<i4>
  %in_reset = arc.root_input "reset", %arg0 : (!arc.storage) -> !arc.state<i1>
  arc.passthrough {
    %3 = arc.state_read %1 : <i4>
    arc.state_write %out_out0 = %3 : <i4>
    %4 = arc.state_read %2 : <i4>
    arc.state_write %out_out1 = %4 : <i4>
  }
  %out_out0 = arc.root_output "out0", %arg0 : (!arc.storage) -> !arc.state<i4>
  %out_out1 = arc.root_output "out1", %arg0 : (!arc.storage) -> !arc.state<i4>
  %0 = arc.state_read %in_clock : <i1>
  arc.clock_tree %0 {
    //  CHECK: [[IN_RESET:%.+]] = arc.state_read %in_reset
    %3 = arc.state_read %in_reset : <i1>
    //  CHECK-NEXT: scf.if [[IN_RESET]] {
    scf.if %3 {
      //   CHECK-NEXT:  arc.state_write [[FOO_ALLOC:%.+]] = %c0_i4
      //   CHECK-NEXT:  arc.state_write [[BAR_ALLOC:%.+]] = %c0_i4
      arc.state_write %1 = %c0_i4 : <i4>
      //   CHECK-NEXT: } else {
    } else {
      // CHECK-NEXT:  [[IN_I0:%.+]] = arc.state_read %in_i0
      // CHECK-NEXT:  arc.state_write [[FOO_ALLOC]] = [[IN_I0]]
      // CHECK-NEXT:  [[IN_I1:%.+]] = arc.state_read %in_i1
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
  // CHECK-NEXT: [[FOO_ALLOC]] = arc.alloc_state %arg0 {name = "foo"} : (!arc.storage) -> !arc.state<i4>
  // CHECK-NEXT: [[BAR_ALLOC]] = arc.alloc_state %arg0 {name = "bar"} : (!arc.storage) -> !arc.state<i4>
  %1 = arc.alloc_state %arg0 {name = "foo"} : (!arc.storage) -> !arc.state<i4>
  %2 = arc.alloc_state %arg0 {name = "bar"} : (!arc.storage) -> !arc.state<i4>
}

// CHECK-LABEL: arc.model "DontGroupResets"
arc.model "DontGroupResets" {
^bb0(%arg0: !arc.storage):
  %c0_i4 = hw.constant 0 : i4
  %true = hw.constant true
  %in_clock = arc.root_input "clock", %arg0 : (!arc.storage) -> !arc.state<i1>
  %in_i0 = arc.root_input "i0", %arg0 : (!arc.storage) -> !arc.state<i4>
  %in_i1 = arc.root_input "i1", %arg0 : (!arc.storage) -> !arc.state<i4>
  %in_reset0 = arc.root_input "reset0", %arg0 : (!arc.storage) -> !arc.state<i1>
  %in_reset1 = arc.root_input "reset1", %arg0 : (!arc.storage) -> !arc.state<i1>
  arc.passthrough {
    %3 = arc.state_read %1 : <i4>
    arc.state_write %out_out0 = %3 : <i4>
    %4 = arc.state_read %2 : <i4>
    arc.state_write %out_out1 = %4 : <i4>
  }
  %out_out0 = arc.root_output "out0", %arg0 : (!arc.storage) -> !arc.state<i4>
  %out_out1 = arc.root_output "out1", %arg0 : (!arc.storage) -> !arc.state<i4>
  %0 = arc.state_read %in_clock : <i1>
  arc.clock_tree %0 {
    //  CHECK: [[IN_RESET0:%.+]] = arc.state_read %in_reset0
    %3 = arc.state_read %in_reset0 : <i1>
    //  CHECK: [[IN_RESET1:%.+]] = arc.state_read %in_reset1
    %4 = arc.state_read %in_reset1 : <i1>
    //  CHECK-NEXT: scf.if [[IN_RESET0]] {
    scf.if %3 {
      //   CHECK-NEXT:  arc.state_write [[FOO_ALLOC:%.+]] = %c0_i4
      arc.state_write %1 = %c0_i4 : <i4>
      //   CHECK-NEXT: } else {
    } else {
      // CHECK-NEXT:  [[IN_I0:%.+]] = arc.state_read %in_i0
      // CHECK-NEXT:  arc.state_write [[FOO_ALLOC]] = [[IN_I0]]
      %5 = arc.state_read %in_i0 : <i4>
      arc.state_write %1 = %5 : <i4>
      // CHECK-NEXT: }
    }
    //  CHECK-NEXT: scf.if [[IN_RESET1]] {
    scf.if %4 {
      //   CHECK-NEXT:  arc.state_write [[BAR_ALLOC:%.+]] = %c0_i4
      arc.state_write %2 = %c0_i4 : <i4>
      //   CHECK-NEXT: } else {
    } else {
      // CHECK-NEXT:  [[IN_I1:%.+]] = arc.state_read %in_i1
      // CHECK-NEXT:  arc.state_write [[BAR_ALLOC]] = [[IN_I1]]
      %6 = arc.state_read %in_i1 : <i4>
      arc.state_write %2 = %6 : <i4>
    }
    // CHECK-NEXT: }
  // CHECK-NEXT: }
  }
  // CHECK-NEXT: [[FOO_ALLOC]] = arc.alloc_state %arg0 {name = "foo"} : (!arc.storage) -> !arc.state<i4>
  // CHECK-NEXT: [[BAR_ALLOC]] = arc.alloc_state %arg0 {name = "bar"} : (!arc.storage) -> !arc.state<i4>
  %1 = arc.alloc_state %arg0 {name = "foo"} : (!arc.storage) -> !arc.state<i4>
  %2 = arc.alloc_state %arg0 {name = "bar"} : (!arc.storage) -> !arc.state<i4>
}

// CHECK-LABEL: arc.model "JustGroupEnables"
arc.model "JustGroupEnables" {
^bb0(%arg0: !arc.storage):
  %c0_i4 = hw.constant 0 : i4
  %true = hw.constant true
  %in_clock = arc.root_input "clock", %arg0 : (!arc.storage) -> !arc.state<i1>
  %in_i0 = arc.root_input "i0", %arg0 : (!arc.storage) -> !arc.state<i4>
  %in_i1 = arc.root_input "i1", %arg0 : (!arc.storage) -> !arc.state<i4>
  %in_en = arc.root_input "en", %arg0 : (!arc.storage) -> !arc.state<i1>
  arc.passthrough {
    %3 = arc.state_read %1 : <i4>
    arc.state_write %out_out0 = %3 : <i4>
    %4 = arc.state_read %2 : <i4>
    arc.state_write %out_out1 = %4 : <i4>
  }
  %out_out0 = arc.root_output "out0", %arg0 : (!arc.storage) -> !arc.state<i4>
  %out_out1 = arc.root_output "out1", %arg0 : (!arc.storage) -> !arc.state<i4>
  %0 = arc.state_read %in_clock : <i1>
  arc.clock_tree %0 {
    //  CHECK: [[IN_EN:%.+]] = arc.state_read %in_en
    %3 = arc.state_read %in_en : <i1>
    //   CHECK-NEXT: arc.state_write [[FOO_ALLOC:%.+]] = %c0_i4
    //   CHECK-NEXT: arc.state_write [[BAR_ALLOC:%.+]] = %c0_i4
    arc.state_write %1 = %c0_i4 : <i4>
    arc.state_write %2 = %c0_i4 : <i4>
    // CHECK-NEXT:   [[IN_I0:%.+]] = arc.state_read %in_i0
    // CHECK-NEXT:   [[IN_I1:%.+]] = arc.state_read %in_i1
    // CHECK-NEXT:   scf.if [[IN_EN]] {
    // CHECK-NEXT:    arc.state_write [[FOO_ALLOC]] = [[IN_I0]]
    // CHECK-NEXT:    arc.state_write [[BAR_ALLOC]] = [[IN_I1]]
    %4 = arc.state_read %in_i0 : <i4>
    arc.state_write %1 = %4 if %3 : <i4>
    %5 = arc.state_read %in_i1 : <i4>
    arc.state_write %2 = %5 if %3 : <i4>
    // CHECK-NEXT:  }
  // CHECK-NEXT: }
  }
  // CHECK-NEXT: [[FOO_ALLOC]] = arc.alloc_state %arg0 {name = "foo"} : (!arc.storage) -> !arc.state<i4>
  // CHECK-NEXT: [[BAR_ALLOC]] = arc.alloc_state %arg0 {name = "bar"} : (!arc.storage) -> !arc.state<i4>
  %1 = arc.alloc_state %arg0 {name = "foo"} : (!arc.storage) -> !arc.state<i4>
  %2 = arc.alloc_state %arg0 {name = "bar"} : (!arc.storage) -> !arc.state<i4>
}

// CHECK-LABEL: arc.model "DontGroupEnables"
arc.model "DontGroupEnables" {
^bb0(%arg0: !arc.storage):
  %c0_i4 = hw.constant 0 : i4
  %true = hw.constant true
  %in_clock = arc.root_input "clock", %arg0 : (!arc.storage) -> !arc.state<i1>
  %in_i0 = arc.root_input "i0", %arg0 : (!arc.storage) -> !arc.state<i4>
  %in_i1 = arc.root_input "i1", %arg0 : (!arc.storage) -> !arc.state<i4>
  %in_en0 = arc.root_input "en0", %arg0 : (!arc.storage) -> !arc.state<i1>
  %in_en1 = arc.root_input "en1", %arg0 : (!arc.storage) -> !arc.state<i1>
  arc.passthrough {
    %3 = arc.state_read %1 : <i4>
    arc.state_write %out_out0 = %3 : <i4>
    %4 = arc.state_read %2 : <i4>
    arc.state_write %out_out1 = %4 : <i4>
  }
  %out_out0 = arc.root_output "out0", %arg0 : (!arc.storage) -> !arc.state<i4>
  %out_out1 = arc.root_output "out1", %arg0 : (!arc.storage) -> !arc.state<i4>
  %0 = arc.state_read %in_clock : <i1>
  arc.clock_tree %0 {
    //  CHECK: [[IN_EN0:%.+]] = arc.state_read %in_en0
    %3 = arc.state_read %in_en0 : <i1>
    //  CHECK: [[IN_EN1:%.+]] = arc.state_read %in_en1
    %4 = arc.state_read %in_en1 : <i1>
    //   CHECK-NEXT: arc.state_write [[FOO_ALLOC:%.+]] = %c0_i4
    //   CHECK-NEXT: arc.state_write [[BAR_ALLOC:%.+]] = %c0_i4
    arc.state_write %1 = %c0_i4 : <i4>
    arc.state_write %2 = %c0_i4 : <i4>
    // CHECK-NEXT:   [[IN_I0:%.+]] = arc.state_read %in_i0
    // CHECK-NEXT:   arc.state_write [[FOO_ALLOC]] = [[IN_I0]] if [[IN_EN0]]
    // CHECK-NEXT:   [[IN_I1:%.+]] = arc.state_read %in_i1
    // CHECK-NEXT:   arc.state_write [[BAR_ALLOC]] = [[IN_I1]] if [[IN_EN1]]
    %5 = arc.state_read %in_i0 : <i4>
    arc.state_write %1 = %5 if %3 : <i4>
    %6 = arc.state_read %in_i1 : <i4>
    arc.state_write %2 = %6 if %4 : <i4>
  // CHECK-NEXT: }
  }
  // CHECK-NEXT: [[FOO_ALLOC]] = arc.alloc_state %arg0 {name = "foo"} : (!arc.storage) -> !arc.state<i4>
  // CHECK-NEXT: [[BAR_ALLOC]] = arc.alloc_state %arg0 {name = "bar"} : (!arc.storage) -> !arc.state<i4>
  %1 = arc.alloc_state %arg0 {name = "foo"} : (!arc.storage) -> !arc.state<i4>
  %2 = arc.alloc_state %arg0 {name = "bar"} : (!arc.storage) -> !arc.state<i4>
}

// CHECK-LABEL: arc.model "GroupEnablesInReset"
arc.model "GroupEnablesInReset" {
^bb0(%arg0: !arc.storage):
  %c0_i4 = hw.constant 0 : i4
  %true = hw.constant true
  %in_clock = arc.root_input "clock", %arg0 : (!arc.storage) -> !arc.state<i1>
  %in_i0 = arc.root_input "i0", %arg0 : (!arc.storage) -> !arc.state<i4>
  %in_i1 = arc.root_input "i1", %arg0 : (!arc.storage) -> !arc.state<i4>
  %in_reset = arc.root_input "reset", %arg0 : (!arc.storage) -> !arc.state<i1>
  %in_en1 = arc.root_input "en1", %arg0 : (!arc.storage) -> !arc.state<i1>
  arc.passthrough {
    %3 = arc.state_read %1 : <i4>
    arc.state_write %out_out0 = %3 : <i4>
    %4 = arc.state_read %2 : <i4>
    arc.state_write %out_out1 = %4 : <i4>
  }
  %out_out0 = arc.root_output "out0", %arg0 : (!arc.storage) -> !arc.state<i4>
  %out_out1 = arc.root_output "out1", %arg0 : (!arc.storage) -> !arc.state<i4>
  %0 = arc.state_read %in_clock : <i1>
  arc.clock_tree %0 {
    //  CHECK: [[IN_EN:%.+]] = arc.state_read %in_en1
    %3 = arc.state_read %in_en1 : <i1>
    //  CHECK-NEXT: [[IN_RESET:%.+]] = arc.state_read %in_reset
    %4 = arc.state_read %in_reset : <i1>
    //  CHECK-NEXT: scf.if [[IN_RESET]] {
    scf.if %4 {
      //   CHECK-NEXT: arc.state_write [[FOO_ALLOC:%.+]] = %c0_i4
      //   CHECK-NEXT: arc.state_write [[BAR_ALLOC:%.+]] = %c0_i4
      arc.state_write %1 = %c0_i4 : <i4>
      arc.state_write %2 = %c0_i4 : <i4>
      //   CHECK-NEXT: } else {
    } else {
      // CHECK-NEXT:   [[IN_I0:%.+]] = arc.state_read %in_i0
      // CHECK-NEXT:   [[IN_I1:%.+]] = arc.state_read %in_i1
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
  // CHECK-NEXT: [[FOO_ALLOC]] = arc.alloc_state %arg0 {name = "foo"} : (!arc.storage) -> !arc.state<i4>
  // CHECK-NEXT: [[BAR_ALLOC]] = arc.alloc_state %arg0 {name = "bar"} : (!arc.storage) -> !arc.state<i4>
  %1 = arc.alloc_state %arg0 {name = "foo"} : (!arc.storage) -> !arc.state<i4>
  %2 = arc.alloc_state %arg0 {name = "bar"} : (!arc.storage) -> !arc.state<i4>
}

// CHECK-LABEL: arc.model "GroupEnablesAndResets"
arc.model "GroupEnablesAndResets" {
^bb0(%arg0: !arc.storage):
  %c0_i4 = hw.constant 0 : i4
  %true = hw.constant true
  %in_clock = arc.root_input "clock", %arg0 : (!arc.storage) -> !arc.state<i1>
  %in_i0 = arc.root_input "i0", %arg0 : (!arc.storage) -> !arc.state<i4>
  %in_i1 = arc.root_input "i1", %arg0 : (!arc.storage) -> !arc.state<i4>
  %in_reset = arc.root_input "reset", %arg0 : (!arc.storage) -> !arc.state<i1>
  %in_en = arc.root_input "en", %arg0 : (!arc.storage) -> !arc.state<i1>
  arc.passthrough {
    %3 = arc.state_read %1 : <i4>
    arc.state_write %out_out0 = %3 : <i4>
    %4 = arc.state_read %2 : <i4>
    arc.state_write %out_out1 = %4 : <i4>
  }
  %out_out0 = arc.root_output "out0", %arg0 : (!arc.storage) -> !arc.state<i4>
  %out_out1 = arc.root_output "out1", %arg0 : (!arc.storage) -> !arc.state<i4>
  %0 = arc.state_read %in_clock : <i1>
  arc.clock_tree %0 {
    //  CHECK: [[IN_RESET:%.+]] = arc.state_read %in_reset
    %3 = arc.state_read %in_reset : <i1>
    //  CHECK: [[IN_EN:%.+]] = arc.state_read %in_en
    %4 = arc.state_read %in_en : <i1>
    //  CHECK-NEXT: scf.if [[IN_RESET]] {
    scf.if %3 {
      //   CHECK-NEXT:  arc.state_write [[FOO_ALLOC:%.+]] = %c0_i4
      //   CHECK-NEXT:  arc.state_write [[BAR_ALLOC:%.+]] = %c0_i4
      arc.state_write %1 = %c0_i4 : <i4>
      //   CHECK-NEXT: } else {
    } else {
      // CHECK-NEXT:   [[IN_I0:%.+]] = arc.state_read %in_i0
      // CHECK-NEXT:   [[IN_I1:%.+]] = arc.state_read %in_i1
      // CHECK-NEXT:   scf.if [[IN_EN]] {
      // CHECK-NEXT:    arc.state_write [[FOO_ALLOC]] = [[IN_I0]]
      // CHECK-NEXT:    arc.state_write [[BAR_ALLOC]] = [[IN_I1]]
      // CHECK-NEXT:   }
      %5 = arc.state_read %in_i0 : <i4>
      arc.state_write %1 = %5 if %4 : <i4>
      // CHECK-NEXT: }
    }
    scf.if %3 {
      arc.state_write %2 = %c0_i4 : <i4>
    } else {
      %6 = arc.state_read %in_i1 : <i4>
      arc.state_write %2 = %6 if %4 : <i4>
    }
    // CHECK-NEXT: }
  }
  // CHECK-NEXT: [[FOO_ALLOC]] = arc.alloc_state %arg0 {name = "foo"} : (!arc.storage) -> !arc.state<i4>
  // CHECK-NEXT: [[BAR_ALLOC]] = arc.alloc_state %arg0 {name = "bar"} : (!arc.storage) -> !arc.state<i4>
  %1 = arc.alloc_state %arg0 {name = "foo"} : (!arc.storage) -> !arc.state<i4>
  %2 = arc.alloc_state %arg0 {name = "bar"} : (!arc.storage) -> !arc.state<i4>
}

// CHECK-LABEL: arc.model "GroupSeparatedResets"
arc.model "GroupSeparatedResets" {
^bb0(%arg0: !arc.storage):
  %c0_i4 = hw.constant 0 : i4
  %true = hw.constant true
  %in_clock = arc.root_input "clock", %arg0 : (!arc.storage) -> !arc.state<i1>
  %in_i0 = arc.root_input "i0", %arg0 : (!arc.storage) -> !arc.state<i4>
  %in_i1 = arc.root_input "i1", %arg0 : (!arc.storage) -> !arc.state<i4>
  %in_reset = arc.root_input "reset", %arg0 : (!arc.storage) -> !arc.state<i1>
  %in_en0 = arc.root_input "en0", %arg0 : (!arc.storage) -> !arc.state<i1>
  %in_en1 = arc.root_input "en1", %arg0 : (!arc.storage) -> !arc.state<i1>
  arc.passthrough {
    %3 = arc.state_read %1 : <i4>
    arc.state_write %out_out0 = %3 : <i4>
    %4 = arc.state_read %2 : <i4>
    arc.state_write %out_out1 = %4 : <i4>
  }
  %out_out0 = arc.root_output "out0", %arg0 : (!arc.storage) -> !arc.state<i4>
  %out_out1 = arc.root_output "out1", %arg0 : (!arc.storage) -> !arc.state<i4>
  %0 = arc.state_read %in_clock : <i1>
  arc.clock_tree %0 {
    //  CHECK: [[IN_RESET:%.+]] = arc.state_read %in_reset
    %3 = arc.state_read %in_reset : <i1>
    //  CHECK: [[IN_EN0:%.+]] = arc.state_read %in_en0
    %4 = arc.state_read %in_en0 : <i1>
    //  CHECK: [[IN_EN1:%.+]] = arc.state_read %in_en1
    //  CHECK-NEXT: scf.if [[IN_RESET]] {
    scf.if %3 {
      //   CHECK-NEXT:  arc.state_write [[FOO_ALLOC:%.+]] = %c0_i4
      //   CHECK-NEXT:  arc.state_write [[BAR_ALLOC:%.+]] = %c0_i4
      arc.state_write %1 = %c0_i4 : <i4>
      //   CHECK-NEXT: } else {
    } else {
      // CHECK-NEXT:  [[IN_I0:%.+]] = arc.state_read %in_i0
      // CHECK-NEXT:  arc.state_write [[FOO_ALLOC]] = [[IN_I0]] if [[IN_EN0]]
      // CHECK-NEXT:  [[IN_I1:%.+]] = arc.state_read %in_i1
      // CHECK-NEXT:  arc.state_write [[BAR_ALLOC]] = [[IN_I1]] if [[IN_EN1]]
      %5 = arc.state_read %in_i0 : <i4>
      arc.state_write %1 = %5 if %4 : <i4>
      // CHECK-NEXT: }
    }
    %6 = arc.state_read %in_en1 : <i1>
    scf.if %3 {
      arc.state_write %2 = %c0_i4 : <i4>
    } else {
      %7 = arc.state_read %in_i1 : <i4>
      arc.state_write %2 = %7 if %6 : <i4>
    }
    // CHECK-NEXT: }
  }
  // CHECK-NEXT: [[FOO_ALLOC]] = arc.alloc_state %arg0 {name = "foo"} : (!arc.storage) -> !arc.state<i4>
  // CHECK-NEXT: [[BAR_ALLOC]] = arc.alloc_state %arg0 {name = "bar"} : (!arc.storage) -> !arc.state<i4>
  %1 = arc.alloc_state %arg0 {name = "foo"} : (!arc.storage) -> !arc.state<i4>
  %2 = arc.alloc_state %arg0 {name = "bar"} : (!arc.storage) -> !arc.state<i4>
}
