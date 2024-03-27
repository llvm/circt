// RUN: arcilator %s --run --jit-entry=main | FileCheck %s
// REQUIRES: arcilator-jit

// CHECK: counter_value = 0
// CHECK-NEXT: counter_value = 1
// CHECK-NEXT: counter_value = 2
// CHECK-NEXT: counter_value = 3
// CHECK-NEXT: counter_value = 4
// CHECK-NEXT: counter_value = 5
// CHECK-NEXT: counter_value = 6
// CHECK-NEXT: counter_value = 7
// CHECK-NEXT: counter_value = 8
// CHECK-NEXT: counter_value = 9
// CHECK-NEXT: counter_value = a

hw.module @counter(in %clk: i1, out o: i8) {
  %seq_clk = seq.to_clock %clk

  %reg = seq.compreg %added, %seq_clk : i8

  %one = hw.constant 1 : i8
  %added = comb.add %reg, %one : i8

  hw.output %reg : i8
}

func.func @main() {
  %zero = arith.constant 0 : i1
  %one = arith.constant 1 : i1
  %lb = arith.constant 0 : index
  %ub = arith.constant 10 : index
  %step = arith.constant 1 : index

  arc.sim.instantiate @counter as %model {
    %init_val = arc.sim.get_port %model, "o" : i8, !arc.sim.instance<@counter>
    arc.sim.emit "counter_value", %init_val : i8

    scf.for %i = %lb to %ub step %step {
      arc.sim.set_input %model, "clk" = %one : i1, !arc.sim.instance<@counter>
      arc.sim.step %model : !arc.sim.instance<@counter>
      arc.sim.set_input %model, "clk" = %zero : i1, !arc.sim.instance<@counter>
      arc.sim.step %model : !arc.sim.instance<@counter>

      %counter_val = arc.sim.get_port %model, "o" : i8, !arc.sim.instance<@counter>
      arc.sim.emit "counter_value", %counter_val : i8
    }
  }

  return
}
