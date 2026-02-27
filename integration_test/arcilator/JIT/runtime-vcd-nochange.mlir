// RUN: arcilator %s --run --jit-entry=main --jit-vcd-file=%t && cat %t | FileCheck %s --match-full-lines --check-prefix VCD
// REQUIRES: arcilator-jit

hw.module @dut(out dout : i136) {
  %cst = hw.constant 0 : i136
  hw.output %cst : i136
}

func.func @main() {
  %inc = arith.constant 1 : i64
  %0 = arith.constant 0 : i64
  arc.sim.instantiate @dut as %model {
    %1 = arith.addi %0, %inc : i64
    arc.sim.set_time %model, %1 : !arc.sim.instance<@dut>
    arc.sim.step %model : !arc.sim.instance<@dut>
    %2 = arith.addi %1, %inc : i64
    arc.sim.set_time %model, %2 : !arc.sim.instance<@dut>
    arc.sim.step %model : !arc.sim.instance<@dut>
    %3 = arith.addi %2, %inc : i64
    arc.sim.set_time %model, %3 : !arc.sim.instance<@dut>
    arc.sim.step %model : !arc.sim.instance<@dut>
    %4 = arith.addi %3, %inc : i64
    arc.sim.set_time %model, %4 : !arc.sim.instance<@dut>
    arc.sim.step %model : !arc.sim.instance<@dut>
    %5 = arith.addi %4, %inc : i64
    arc.sim.set_time %model, %5 : !arc.sim.instance<@dut>
    arc.sim.step %model : !arc.sim.instance<@dut>
    %6 = arith.addi %5, %inc : i64
    arc.sim.set_time %model, %6 : !arc.sim.instance<@dut>
    arc.sim.step %model : !arc.sim.instance<@dut>
    %7 = arith.addi %6, %inc : i64
    arc.sim.set_time %model, %7 : !arc.sim.instance<@dut>
    arc.sim.step %model : !arc.sim.instance<@dut>
    %8 = arith.addi %7, %inc : i64
    arc.sim.set_time %model, %8 : !arc.sim.instance<@dut>
    arc.sim.step %model : !arc.sim.instance<@dut>
    %9 = arith.addi %8, %inc : i64
    arc.sim.set_time %model, %9 : !arc.sim.instance<@dut>
  }
  return
}

// VCD-LABEL: $version
// VCD-NEXT:      Some cryptic ArcRuntime magic
// VCD-NEXT:  $end
// VCD-NEXT:  $timescale 1fs $end
// VCD-NEXT:  $scope module dut $end
// VCD-NEXT:   $var wire 136 ! dout $end
// VCD-NEXT:  $upscope $end
// VCD-NEXT:  $enddefinitions $end
// VCD-NEXT:  #0
// VCD-NEXT:  b0000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000 !
// VCD-NEXT:  #9
