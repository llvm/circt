// RUN: arcilator %s --run --jit-entry=main | FileCheck %s
// RUN: arcilator %s --observe-wires --observe-registers --observe-ports --trace-taps --run --jit-entry=main --jit-vcd | FileCheck %s
// RUN: arcilator %s --observe-wires --observe-registers --observe-ports --trace-taps --run --jit-entry=main --jit-vcd && cat shiftreg.vcd | FileCheck %s --match-full-lines --check-prefix VCD
// REQUIRES: arcilator-jit

// CHECK-LABEL: output = ca
// CHECK-NEXT: output = ca
// CHECK-NEXT: output = 0
// CHECK-NEXT: output = fe
// CHECK-NEXT: output = ff

module {

  hw.module @shiftreg(in %clock : i1, in %reset : i1, in %en : i1, in %din : i8, out dout : i8) {
    %seq_clk = seq.to_clock %clock
    %srA = seq.firreg %0 clock %seq_clk preset 0xFE : i8
    %srB = seq.firreg %1 clock %seq_clk : i8
    %srC = seq.firreg %2 clock %seq_clk preset 0xCA : i8
    %0 = comb.mux bin %en, %din, %srA : i8
    %1 = comb.mux bin %en, %srA, %srB : i8
    %2 = comb.mux bin %en, %srB, %srC : i8
    hw.output %srC : i8
  }

  func.func public @main() {
    %ff = arith.constant 0xFF : i8
    %false = arith.constant 0 : i1
    %true = arith.constant 1 : i1

    arc.sim.instantiate @shiftreg as %model {
      arc.sim.step %model : !arc.sim.instance<@shiftreg>
      arc.sim.set_input %model, "en" = %false : i1, !arc.sim.instance<@shiftreg>
      arc.sim.set_input %model, "reset" = %false : i1, !arc.sim.instance<@shiftreg>
      arc.sim.set_input %model, "din" = %ff : i8, !arc.sim.instance<@shiftreg>

      %res0 = arc.sim.get_port %model, "dout" : i8, !arc.sim.instance<@shiftreg>
      arc.sim.emit "output", %res0 : i8

      arc.sim.set_input %model, "clock" = %true : i1, !arc.sim.instance<@shiftreg>
      arc.sim.step %model : !arc.sim.instance<@shiftreg>
      arc.sim.set_input %model, "clock" = %false : i1, !arc.sim.instance<@shiftreg>
      arc.sim.step %model : !arc.sim.instance<@shiftreg>

      %res1 = arc.sim.get_port %model, "dout" : i8, !arc.sim.instance<@shiftreg>
      arc.sim.emit "output", %res1 : i8

      arc.sim.set_input %model, "en" = %true : i1, !arc.sim.instance<@shiftreg>

      arc.sim.set_input %model, "clock" = %true : i1, !arc.sim.instance<@shiftreg>
      arc.sim.step %model : !arc.sim.instance<@shiftreg>
      arc.sim.set_input %model, "clock" = %false : i1, !arc.sim.instance<@shiftreg>
      arc.sim.step %model : !arc.sim.instance<@shiftreg>
      %res2 = arc.sim.get_port %model, "dout" : i8, !arc.sim.instance<@shiftreg>
      arc.sim.emit "output", %res2 : i8

      arc.sim.set_input %model, "clock" = %true : i1, !arc.sim.instance<@shiftreg>
      arc.sim.step %model : !arc.sim.instance<@shiftreg>
      arc.sim.set_input %model, "clock" = %false : i1, !arc.sim.instance<@shiftreg>
      arc.sim.step %model : !arc.sim.instance<@shiftreg>
      %res3 = arc.sim.get_port %model, "dout" : i8, !arc.sim.instance<@shiftreg>
      arc.sim.emit "output", %res3 : i8

      arc.sim.set_input %model, "clock" = %true : i1, !arc.sim.instance<@shiftreg>
      arc.sim.step %model : !arc.sim.instance<@shiftreg>
      arc.sim.set_input %model, "clock" = %false : i1, !arc.sim.instance<@shiftreg>
      arc.sim.step %model : !arc.sim.instance<@shiftreg>
      %res4 = arc.sim.get_port %model, "dout" : i8, !arc.sim.instance<@shiftreg>
      arc.sim.emit "output", %res4 : i8
    }
    return
  }
}

// VCD:      $date
// VCD-NEXT:     October 21, 2015
// VCD-NEXT: $end
// VCD-NEXT: $version
// VCD-NEXT:     Some cryptic JIT MLIR magic
// VCD-NEXT: $end
// VCD-NEXT: $timescale 1ns $end
// VCD-NEXT: $scope module shiftreg $end
// VCD-NEXT:  $var wire 1 ! clock $end
// VCD-NEXT:  $var wire 8 $ din $end
// VCD-NEXT:  $var wire 8 ' dout $end
// VCD-NEXT:  $var wire 1 # en $end
// VCD-NEXT:  $var wire 1 " reset $end
// VCD-NEXT:  $var wire 8 ( srA $end
// VCD-NEXT:  $var wire 8 & srB $end
// VCD-NEXT:  $var wire 8 % srC $end
// VCD-NEXT: $upscope $end
// VCD-NEXT: $enddefinitions $end
// VCD-NEXT: #0
// VCD-NEXT: 0!
// VCD-NEXT: 0"
// VCD-NEXT: 0#
// VCD-NEXT: b00000000 $
// VCD-NEXT: b00000000 %
// VCD-NEXT: b00000000 &
// VCD-NEXT: b00000000 '
// VCD-NEXT: b00000000 (
// VCD-NEXT: #1
// VCD-NEXT: b11111110 (
// VCD-NEXT: b11001010 %
// VCD-NEXT: #2
// VCD-NEXT: b11001010 '
// VCD-NEXT: #3
// VCD-NEXT: 1!
// VCD-NEXT: b11111111 $
// VCD-NEXT: #4
// VCD-NEXT: 0!
// VCD-NEXT: #5
// VCD-NEXT: 1!
// VCD-NEXT: 1#
// VCD-NEXT: b00000000 %
// VCD-NEXT: b00000000 '
// VCD-NEXT: b11111111 (
// VCD-NEXT: b11111110 &
// VCD-NEXT: #6
// VCD-NEXT: 0!
// VCD-NEXT: #7
// VCD-NEXT: 1!
// VCD-NEXT: b11111110 %
// VCD-NEXT: b11111110 '
// VCD-NEXT: b11111111 &
// VCD-NEXT: #8
// VCD-NEXT: 0!
// VCD-NEXT: #9
// VCD-NEXT: 1!
// VCD-NEXT: b11111111 %
// VCD-NEXT: b11111111 '
// VCD-NEXT: #10
// VCD-NEXT: 0!
// VCD-NEXT: #11
