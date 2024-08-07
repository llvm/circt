// RUN: arcilator %s --run --jit-entry=main | FileCheck %s
// REQUIRES: arcilator-jit

// CHECK:      c = 0
// CHECK-NEXT: c = 5
sim.func.dpi @dpi(in %a : i32, in %b : i32, out c : i32) attributes {verilogName = "adder_func"}
func.func @adder_func(%arg0: i32, %arg1: i32, %arg2: !llvm.ptr) {
  %0 = arith.addi %arg0, %arg1 : i32
  llvm.store %0, %arg2 : i32, !llvm.ptr
  return
}
hw.module @adder(in %clock : i1, in %a : i32, in %b : i32, out c : i32) {
  %seq_clk = seq.to_clock %clock

  %0 = sim.func.call @dpi(%a, %b) clock %seq_clk : (i32, i32) -> i32
  hw.output %0 : i32
}
func.func @main() {
  %c2_i32 = arith.constant 2 : i32
  %c3_i32 = arith.constant 3 : i32
  %one = arith.constant 1 : i1
  %zero = arith.constant 0 : i1
  arc.sim.instantiate @adder as %arg0 {
    arc.sim.set_input %arg0, "a" = %c2_i32 : i32, !arc.sim.instance<@adder>
    arc.sim.set_input %arg0, "b" = %c3_i32 : i32, !arc.sim.instance<@adder>
    arc.sim.set_input %arg0, "clock" = %one : i1, !arc.sim.instance<@adder>

    arc.sim.step %arg0 : !arc.sim.instance<@adder>
    arc.sim.set_input %arg0, "clock" = %zero : i1, !arc.sim.instance<@adder>
    %0 = arc.sim.get_port %arg0, "c" : i32, !arc.sim.instance<@adder>
    arc.sim.emit "c", %0 : i32

    arc.sim.step %arg0 : !arc.sim.instance<@adder>
    arc.sim.set_input %arg0, "clock" = %one : i1, !arc.sim.instance<@adder>
    %2 = arc.sim.get_port %arg0, "c" : i32, !arc.sim.instance<@adder>
    arc.sim.emit "c", %2 : i32
  }
  return
}
