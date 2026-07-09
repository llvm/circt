// RUN: arcilator %s --run --jit-entry=main --jit-vcd-file=trace.vcd
// RUN: FileCheck %s --input-file=TestWorkDir0%{fs-sep}trace.vcd --check-prefixes=CHECK,VCD1
// RUN: FileCheck %s --input-file=TestWorkDir1%{fs-sep}trace.vcd --check-prefixes=CHECK,VCD2
// RUN: FileCheck %s --input-file=overriddenFileName.vcd --check-prefixes=CHECK,VCD3
// REQUIRES: arcilator-jit

hw.module @dut(in %in : i2, out o : i2) {
  hw.output %in : i2
}

func.func @main() {
  %inc = arith.constant 1 : i64
  %one    = arith.constant 1 : i2
  %two    = arith.constant 2 : i2
  %three  = arith.constant 3 : i2

  arc.sim.instantiate @dut as %model runtime ("workDir=TestWorkDir0") {
    arc.sim.set_time %model, %inc : !arc.sim.instance<@dut>
    arc.sim.set_input %model, "in" = %one : i2, !arc.sim.instance<@dut>
    arc.sim.step %model by %inc : !arc.sim.instance<@dut>
  }

  arc.sim.instantiate @dut as %model runtime ("workDir=TestWorkDir1") {
    arc.sim.set_time %model, %inc : !arc.sim.instance<@dut>
    arc.sim.set_input %model, "in" = %two : i2, !arc.sim.instance<@dut>
    arc.sim.step %model by %inc : !arc.sim.instance<@dut>
  }

  arc.sim.instantiate @dut as %model runtime ("traceFile=overriddenFileName.vcd") {
    arc.sim.set_time %model, %inc : !arc.sim.instance<@dut>
    arc.sim.set_input %model, "in" = %three : i2, !arc.sim.instance<@dut>
    arc.sim.step %model by %inc : !arc.sim.instance<@dut>
  }

  return
}

// CHECK-LABEL: #1
// VCD1-NEXT:   b01
// VCD2-NEXT:   b10
// VCD3-NEXT:   b11
