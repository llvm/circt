// RUN: arcilator %s --run --jit-entry=main | FileCheck --match-full-lines %s
// REQUIRES: arcilator-jit
// XFAIL: *

// TODO: This test is currently expected to fail because llhd.constant_time
// is not being lowered. We need to implement proper time conversion from
// llhd.time to i64 femtoseconds.

// Test a simple process that waits and then halts
// The process should output 0 initially, then 42 after one time step

// CHECK:      process_output = 00000000
// CHECK-NEXT: process_output = 0000002a

hw.module @ProcessTest(out x : i32) {
  %time = llhd.constant_time <1ns, 0d, 0e>
  %c0_i32 = hw.constant 0 : i32
  %c42_i32 = hw.constant 42 : i32

  %proc_result = llhd.process -> i32 {
    llhd.wait yield (%c0_i32 : i32), delay %time, ^bb1
  ^bb1:
    llhd.halt %c42_i32 : i32
  }

  hw.output %proc_result : i32
}

func.func @main() {
  arc.sim.instantiate @ProcessTest as %model {
    // Get initial output (should be 0)
    %init_val = arc.sim.get_port %model, "x" : i32, !arc.sim.instance<@ProcessTest>
    arc.sim.emit "process_output", %init_val : i32

    // Step the simulation to trigger the process
    arc.sim.step %model : !arc.sim.instance<@ProcessTest>

    // Get output after step (should be 42)
    %final_val = arc.sim.get_port %model, "x" : i32, !arc.sim.instance<@ProcessTest>
    arc.sim.emit "process_output", %final_val : i32
  }

  return
}

