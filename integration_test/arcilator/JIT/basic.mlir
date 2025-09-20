// RUN: arcilator %s --run --jit-entry=main | FileCheck %s
// RUN: arcilator %s --observe-wires --observe-registers --observe-ports --trace-taps --run --jit-entry=main | FileCheck %s
// RUN: arcilator %s --observe-wires --observe-registers --observe-ports --trace-taps --run --jit-entry=main --jit-vcd | FileCheck %s
// RUN: arcilator %s --observe-wires --observe-registers --observe-ports --trace-taps --run --jit-entry=main --jit-vcd && cat adder.vcd | FileCheck %s --match-full-lines --check-prefix VCD
// REQUIRES: arcilator-jit

// CHECK: output = 5

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


// VCD:      $date
// VCD-NEXT:    October 21, 2015
// VCD-NEXT: $end
// VCD-NEXT: $version
// VCD-NEXT:    Some cryptic JIT MLIR magic
// VCD-NEXT: $end
// VCD-NEXT: $timescale 1ns $end
// VCD-NEXT: $scope module adder $end
// VCD-NEXT:  $var wire 8 ! a $end
// VCD-NEXT:  $var wire 8 " b $end
// VCD-NEXT:  $var wire 8 # c $end
// VCD-NEXT: $upscope $end
// VCD-NEXT: $enddefinitions $end
// VCD-NEXT: #0
// VCD-NEXT: b00000000 !
// VCD-NEXT: b00000000 "
// VCD-NEXT: b00000000 #
// VCD-NEXT: #1
// VCD-NEXT: b00000010 !
// VCD-NEXT: b00000011 "
// VCD-NEXT: b00000101 #
// VCD-NEXT: #2
