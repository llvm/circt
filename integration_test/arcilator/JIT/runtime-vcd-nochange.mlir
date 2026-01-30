// RUN: arcilator %s --run --jit-entry=main --jit-vcd-file=%t && cat %t | FileCheck %s --match-full-lines --check-prefix VCD

hw.module @dut(out dout : i136) {
  %cst = hw.constant 0 : i136
  hw.output %cst : i136
}

func.func @main() {
  arc.sim.instantiate @dut as %model {
    arc.sim.step %model : !arc.sim.instance<@dut>
    arc.sim.step %model : !arc.sim.instance<@dut>
    arc.sim.step %model : !arc.sim.instance<@dut>
    arc.sim.step %model : !arc.sim.instance<@dut>
    arc.sim.step %model : !arc.sim.instance<@dut>
    arc.sim.step %model : !arc.sim.instance<@dut>
    arc.sim.step %model : !arc.sim.instance<@dut>
    arc.sim.step %model : !arc.sim.instance<@dut>
  }
  return
}

// VCD-LABEL: $version
// VCD-NEXT:      Some cryptic ArcRuntime magic
// VCD-NEXT:  $end
// VCD-NEXT:  $timescale 1ns $end
// VCD-NEXT:  $scope module dut $end
// VCD-NEXT:   $var wire 136 ! dout $end
// VCD-NEXT:  $upscope $end
// VCD-NEXT:  $enddefinitions $end
// VCD-NEXT:  #0
// VCD-NEXT:  b0000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000 !
// VCD-NEXT:  #9
