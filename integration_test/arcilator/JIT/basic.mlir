// RUN: arcilator %s --run --jit-entry=main | FileCheck %s
// REQUIRES: arcilator-jit

// CHECK: output = 05

hw.module @adder(in %a: i8, in %b: i8, out c: i8) {
  %res = comb.add %a, %b : i8
  hw.output %res : i8
}

func.func @main() {
  %two = arith.constant 2 : i8
  %three = arith.constant 3 : i8

  arc.sim.instantiate @adder as %model {
    arc.sim.set_input %model, "a" = %two : i8, !arc.sim.instance<@adder>
    arc.sim.set_input %model, "b" = %three : i8, !arc.sim.instance<@adder>
    
    arc.sim.step %model : !arc.sim.instance<@adder>
  
    %res = arc.sim.get_port %model, "c" : i8, !arc.sim.instance<@adder>
    arc.sim.emit "output", %res : i8
  }

  return
}
