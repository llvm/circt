// RUN: arcilator %s --run --jit-entry=main | FileCheck %s
// REQUIRES: arcilator-jit

// CHECK:      c = 5

sim.func.dpi @dpi(in %a: i8, in %b: i8, out c: i8) attributes {verilogName="dpi_adder"}

func.func @dpi_adder(%a: i8, %b: i8, %c: !llvm.ptr) {
  %0 = arith.addi %a, %b: i8
  llvm.store %0, %c : i8, !llvm.ptr
  func.return
}

hw.module @adder(in %clock: !seq.clock, in %a: i8, in %b: i8, out c: i8) {
  %dpi_result = sim.func.dpi.call @dpi(%a, %b) clock %clock : (i8, i8) -> (i8)
  hw.output %dpi_result : i8
}

func.func @main() {
  %two = arith.constant 2 : i8
  %three = arith.constant 3 : i8

  arc.sim.instantiate @adder as %model {
    arc.sim.set_input %model, "a" = %two : i8, !arc.sim.instance<@adder>
    arc.sim.set_input %model, "b" = %three : i8, !arc.sim.instance<@adder>
    
    arc.sim.step %model : !arc.sim.instance<@adder>
  
    %c = arc.sim.get_port %model, "c" : i8, !arc.sim.instance<@adder>

    arc.sim.emit "c", %c : i8
  }

  return
}
