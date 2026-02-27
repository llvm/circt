// RUN: arcilator %s --run --jit-entry=main --no-runtime                      | FileCheck --match-full-lines --check-prefixes=CHECK,DBGOFF %s
// RUN: arcilator %s --run --jit-entry=main                                   | FileCheck --match-full-lines --check-prefixes=CHECK,DBGOFF %s
// RUN: arcilator %s --run --jit-entry=main --extra-runtime-args="foo"        | FileCheck --match-full-lines --check-prefixes=CHECK,DBGOFF %s
// RUN: arcilator %s --run --jit-entry=main --extra-runtime-args="debug"      | FileCheck --match-full-lines --check-prefixes=CHECK,DBGON  %s
// RUN: arcilator %s --run --jit-entry=main --extra-runtime-args="foo;debug"  | FileCheck --match-full-lines --check-prefixes=CHECK,DBGON  %s
// REQUIRES: arcilator-jit

// Check that the JIT runtime library allocates state memory and
// optionally emits debug prints.

// DBGOFF-NOT: [ArcRuntime] Created instance of model "flipflop" with ID 0
// DBGOFF-NOT: [ArcRuntime] Instance with ID 0 initialized

// DBGON:      [ArcRuntime] Created instance of model "flipflop" with ID 0
// DBGON:      [ArcRuntime] Instance with ID 0 initialized

// CHECK:      d0 = 00
// CHECK-NEXT: d0 = ca

// DBGOFF-NOT: [ArcRuntime] Deleting instance of model "flipflop" with ID 0 after 1 step(s)
// DBGON-NEXT: [ArcRuntime] Deleting instance of model "flipflop" with ID 0 after 1 step(s)

// DBGOFF-NOT: [ArcRuntime] Created instance of model "flipflop" with ID 1
// DBGOFF-NOT: [ArcRuntime] Instance with ID 1 initialized

// DBGON-NEXT: [ArcRuntime] Created instance of model "flipflop" with ID 1
// DBGON:      [ArcRuntime] Instance with ID 1 initialized

// CHECK:      d1 = 00
// CHECK-NEXT: d1 = fe

// DBGOFF-NOT: [ArcRuntime] Deleting instance of model "flipflop" with ID 1 after 3 step(s)
// DBGON-NEXT: [ArcRuntime] Deleting instance of model "flipflop" with ID 1 after 3 step(s)

hw.module @flipflop(in %clk: i1, in %d: i8, out q: i8) {
  %seq_clk = seq.to_clock %clk
  %reg = seq.compreg %d, %seq_clk : i8
  hw.output %reg : i8
}

func.func @main() {
  %cstCA = arith.constant 0xCA : i8
  %cstFE = arith.constant 0xFE : i8
  %one = arith.constant 1 : i1

  arc.sim.instantiate @flipflop as %model {
    %init_val = arc.sim.get_port %model, "d" : i8, !arc.sim.instance<@flipflop>
    arc.sim.emit "d0", %init_val : i8

    arc.sim.set_input %model, "clk" = %one : i1, !arc.sim.instance<@flipflop>
    arc.sim.set_input %model, "d" = %cstCA : i8, !arc.sim.instance<@flipflop>
    arc.sim.step %model : !arc.sim.instance<@flipflop>

    %step_val = arc.sim.get_port %model, "d" : i8, !arc.sim.instance<@flipflop>
    arc.sim.emit "d0", %step_val : i8
  }

  arc.sim.instantiate @flipflop as %model {
    %init_val = arc.sim.get_port %model, "d" : i8, !arc.sim.instance<@flipflop>
    arc.sim.emit "d1", %init_val : i8

    arc.sim.set_input %model, "clk" = %one : i1, !arc.sim.instance<@flipflop>
    arc.sim.set_input %model, "d" = %cstFE : i8, !arc.sim.instance<@flipflop>
    arc.sim.step %model : !arc.sim.instance<@flipflop>
    arc.sim.step %model : !arc.sim.instance<@flipflop>
    arc.sim.step %model : !arc.sim.instance<@flipflop>

    %step_val = arc.sim.get_port %model, "d" : i8, !arc.sim.instance<@flipflop>
    arc.sim.emit "d1", %step_val : i8
  }

  return
}
