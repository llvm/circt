// RUN: arcilator %s --run --jit-entry=main | FileCheck %s
// REQUIRES: arcilator-jit

// CHECK:      o1 = 02
// CHECK-NEXT: o2 = 05
// CHECK-NEXT: o1 = 03
// CHECK-NEXT: o2 = 06
// CHECK-NEXT: o1 = 04
// CHECK-NEXT: o2 = 07

func.func private @random() -> i32 {
  %0 = arith.constant 2 : i32
  func.return %0: i32
}

hw.module @counter(in %clk: i1, out o1: i8, out o2: i8) {
  %seq_clk = seq.to_clock %clk
  %c0_i8 = hw.constant 0 : i8

  %r0 = seq.compreg %added1, %seq_clk initial %0#0 : i8
  %r1 = seq.compreg %added2, %seq_clk initial %0#1 : i8
  %0:2 = seq.initial () {
    %1 = func.call @random() : () -> i32
    %2 = comb.extract %1 from 0 : (i32) -> i8
    %3 = hw.constant 5 : i8
    seq.yield %2, %3: i8, i8
  } : () -> (!seq.immutable<i8>, !seq.immutable<i8>)

  %one = hw.constant 1 : i8
  %added1 = comb.add %r0, %one : i8
  %added2 = comb.add %r1, %one : i8

  hw.output %r0, %r1 : i8, i8
}

func.func @main() {
  %zero = arith.constant 0 : i1
  %one = arith.constant 1 : i1
  %lb = arith.constant 0 : index
  %ub = arith.constant 2 : index
  %step = arith.constant 1 : index

  arc.sim.instantiate @counter as %model {
    arc.sim.step %model : !arc.sim.instance<@counter>
    %init_val1 = arc.sim.get_port %model, "o1" : i8, !arc.sim.instance<@counter>
    %init_val2 = arc.sim.get_port %model, "o2" : i8, !arc.sim.instance<@counter>

    arc.sim.emit "o1", %init_val1 : i8
    arc.sim.emit "o2", %init_val2 : i8

    scf.for %i = %lb to %ub step %step {
      arc.sim.set_input %model, "clk" = %one : i1, !arc.sim.instance<@counter>
      arc.sim.step %model : !arc.sim.instance<@counter>
      arc.sim.set_input %model, "clk" = %zero : i1, !arc.sim.instance<@counter>
      arc.sim.step %model : !arc.sim.instance<@counter>

      %counter_val1 = arc.sim.get_port %model, "o1" : i8, !arc.sim.instance<@counter>
      arc.sim.emit "o1", %counter_val1 : i8

      %counter_val2 = arc.sim.get_port %model, "o2" : i8, !arc.sim.instance<@counter>
      arc.sim.emit "o2", %counter_val2 : i8

    }
  }

  return
}
