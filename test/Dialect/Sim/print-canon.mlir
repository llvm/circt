// RUN: circt-opt --canonicalize  %s | FileCheck %s

// CHECK-LABEL: hw.module @always_disabled
// CHECK-NOT: sim.print
hw.module @always_disabled(in %clock: !seq.clock) {
  %false = hw.constant false
  %lit = sim.fmt.literal "Foo"
  sim.print %lit on %clock if %false
}

// CHECK-LABEL: hw.module @emtpy_proc_print
// CHECK-NOT: sim.proc.print
hw.module @emtpy_proc_print(in %trigger: i1) {
  hw.triggered posedge %trigger {
    %epsilon = sim.fmt.literal ""
    sim.proc.print %epsilon
  }
}
