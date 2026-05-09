// RUN: arcilator %s --run --jit-entry=main --jit-fst-file=%t && test -s %t
// REQUIRES: arcilator-jit, libfst

hw.module @dut(out dout : i136) {
  %cst = hw.constant 0 : i136
  hw.output %cst : i136
}

func.func @main() {
  %inc = arith.constant 1 : i64
  arc.sim.instantiate @dut as %model {
    arc.sim.set_time %model, %inc : !arc.sim.instance<@dut>
    arc.sim.step %model by %inc : !arc.sim.instance<@dut>
    arc.sim.step %model by %inc : !arc.sim.instance<@dut>
    arc.sim.step %model by %inc : !arc.sim.instance<@dut>
    arc.sim.step %model by %inc : !arc.sim.instance<@dut>
    arc.sim.step %model by %inc : !arc.sim.instance<@dut>
    arc.sim.step %model by %inc : !arc.sim.instance<@dut>
    arc.sim.step %model by %inc : !arc.sim.instance<@dut>
    arc.sim.step %model by %inc : !arc.sim.instance<@dut>
  }
  return
}
