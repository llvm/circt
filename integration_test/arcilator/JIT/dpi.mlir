// RUN: split-file %s %t
// RUN: %host_cc %t/shared_lib.c --shared -o %t/shared_lib.so
// RUN: arcilator %t/dpi.mlir --run --jit-entry=main --shared-libs=%t/shared_lib.so | FileCheck --match-full-lines %s
// REQUIRES: arcilator-jit


//--- shared_lib.c
void mul_shared(int a, int b, int *result) { *result = a * b; }

//--- dpi.mlir
// CHECK:      c = {{0*}}0
// CHECK-NEXT: d = {{0*}}0
// CHECK-NEXT: c = {{0*}}5
// CHECK-NEXT: d = {{0*}}6
sim.func.dpi @mul_shared(in %a : i32, in %b : i32, out c : i32)
sim.func.dpi @add_mlir(in %a : i32, in %b : i32, out c : i32) attributes {verilogName = "add_mlir_impl"}
func.func @add_mlir_impl(%arg0: i32, %arg1: i32, %arg2: !llvm.ptr) {
  %0 = arith.addi %arg0, %arg1 : i32
  llvm.store %0, %arg2 : i32, !llvm.ptr
  return
}

hw.module @arith(in %clock : i1, in %a : i32, in %b : i32, out c : i32, out d : i32) {
  %seq_clk = seq.to_clock %clock
  %0 = sim.func.dpi.call @add_mlir(%a, %b) clock %seq_clk : (i32, i32) -> i32
  %1 = sim.func.dpi.call @mul_shared(%a, %b) clock %seq_clk : (i32, i32) -> i32
  hw.output %0, %1 : i32, i32
}

func.func @main() {
  %c2_i32 = arith.constant 2 : i32
  %c3_i32 = arith.constant 3 : i32
  %one = arith.constant 1 : i1
  %zero = arith.constant 0 : i1
  arc.sim.instantiate @arith as %arg0 {
    arc.sim.set_input %arg0, "a" = %c2_i32 : i32, !arc.sim.instance<@arith>
    arc.sim.set_input %arg0, "b" = %c3_i32 : i32, !arc.sim.instance<@arith>

    arc.sim.set_input %arg0, "clock" = %zero : i1, !arc.sim.instance<@arith>
    arc.sim.step %arg0 : !arc.sim.instance<@arith>
    %0 = arc.sim.get_port %arg0, "c" : i32, !arc.sim.instance<@arith>
    %1 = arc.sim.get_port %arg0, "d" : i32, !arc.sim.instance<@arith>
    arc.sim.emit "c", %0 : i32
    arc.sim.emit "d", %1 : i32

    arc.sim.set_input %arg0, "clock" = %one : i1, !arc.sim.instance<@arith>
    arc.sim.step %arg0 : !arc.sim.instance<@arith>
    %2 = arc.sim.get_port %arg0, "c" : i32, !arc.sim.instance<@arith>
    %3 = arc.sim.get_port %arg0, "d" : i32, !arc.sim.instance<@arith>
    arc.sim.emit "c", %2 : i32
    arc.sim.emit "d", %3 : i32
  }
  return
}
