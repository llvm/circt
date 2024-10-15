// RUN: arcilator --run %s | FileCheck %s
// REQUIRES: arcilator-jit

// CHECK:      0 0 0 0
// CHECK-NEXT: 1 1 1 1
// CHECK-NEXT: 0 0 1 1
// CHECK-NEXT: 1 1 0 1
// CHECK-NEXT: 0 0 0 1
// CHECK-NEXT: 1 1 1 0
// CHECK-NEXT: 0 0 1 0
// CHECK-NEXT: 1 1 0 0
// CHECK-NEXT: 0 0 0 0
// CHECK-NEXT: 1 1 1 1

arc.define @Not(%arg0: i1) -> i1 {
  %true = hw.constant true
  %0 = comb.xor %arg0, %true : i1
  arc.output %0 : i1
}

hw.module @ClockDivBy4(in %clock: !seq.clock, out div1: !seq.clock, out div2: !seq.clock, out div4: !seq.clock) {
  %q0 = arc.state @Not(%q0) clock %clock latency 1 {names = ["q0"]} : (i1) -> i1
  %0 = seq.to_clock %q0
  %q1 = arc.state @Not(%q1) clock %0 latency 1 {names = ["q1"]} : (i1) -> i1
  %1 = seq.to_clock %q1
  hw.output %clock, %0, %1 : !seq.clock, !seq.clock, !seq.clock
}

func.func @entry() {
  arc.sim.instantiate @ClockDivBy4 as %dut {
    arc.sim.step %dut : !arc.sim.instance<@ClockDivBy4>

    %lb = arith.constant 0 : index
    %ub = arith.constant 10 : index
    %step = arith.constant 1 : index
    scf.for %i = %lb to %ub step %step {
      %i0 = index.castu %i : index to i1
      %clock = seq.to_clock %i0

      arc.sim.set_input %dut, "clock" = %clock : !seq.clock, !arc.sim.instance<@ClockDivBy4>
      arc.sim.step %dut : !arc.sim.instance<@ClockDivBy4>

      %0 = arc.sim.get_port %dut, "div1" : !seq.clock, !arc.sim.instance<@ClockDivBy4>
      %1 = arc.sim.get_port %dut, "div2" : !seq.clock, !arc.sim.instance<@ClockDivBy4>
      %2 = arc.sim.get_port %dut, "div4" : !seq.clock, !arc.sim.instance<@ClockDivBy4>

      %3 = llvm.mlir.addressof @string : !llvm.ptr
      %4 = seq.from_clock %0
      %5 = seq.from_clock %1
      %6 = seq.from_clock %2
      %7 = arith.extui %i0 : i1 to i32
      %8 = arith.extui %4 : i1 to i32
      %9 = arith.extui %5 : i1 to i32
      %10 = arith.extui %6 : i1 to i32
      llvm.call @printf(%3, %7, %8, %9, %10) vararg(!llvm.func<i32 (ptr, ...)>) : (!llvm.ptr, i32, i32, i32, i32) -> i32
    }
  }

  return
}

llvm.mlir.global constant @string("%u %u %u %u\n\00") : !llvm.array<13 x i8>

llvm.func @printf(!llvm.ptr, ...) -> i32
