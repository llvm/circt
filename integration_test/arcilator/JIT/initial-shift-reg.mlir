// RUN: arcilator %s --run --jit-entry=main | FileCheck --match-full-lines %s
// RUN: arcilator %s --run --jit-entry=main --jit-vcd-file=%t && cat %t | FileCheck %s --match-full-lines --check-prefix VCD
// REQUIRES: arcilator-jit

// CHECK-LABEL: output = ca
// CHECK-NEXT:  output = ca
// CHECK-NEXT:  output = 00
// CHECK-NEXT:  output = fe
// CHECK-NEXT:  output = ff

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

  func.func @main() {
    %ff = arith.constant 0xFF : i8
    %false = arith.constant 0 : i1
    %true = arith.constant 1 : i1
    %t0 = arith.constant 10 : i64
    %t1 = arith.constant 20 : i64
    %t2 = arith.constant 30 : i64
    %t3 = arith.constant 40 : i64
    %t4 = arith.constant 50 : i64
    %t5 = arith.constant 60 : i64
    %t6 = arith.constant 70 : i64
    %t7 = arith.constant 80 : i64
    %t8 = arith.constant 90 : i64
    %t9 = arith.constant 100 : i64

    arc.sim.instantiate @shiftreg as %model {
      arc.sim.set_time %model, %t0 : !arc.sim.instance<@shiftreg>
      arc.sim.step %model : !arc.sim.instance<@shiftreg>
      arc.sim.set_input %model, "en" = %false : i1, !arc.sim.instance<@shiftreg>
      arc.sim.set_input %model, "reset" = %false : i1, !arc.sim.instance<@shiftreg>
      arc.sim.set_input %model, "din" = %ff : i8, !arc.sim.instance<@shiftreg>

      %res0 = arc.sim.get_port %model, "dout" : i8, !arc.sim.instance<@shiftreg>
      arc.sim.emit "output", %res0 : i8

      arc.sim.set_time %model, %t1 : !arc.sim.instance<@shiftreg>
      arc.sim.set_input %model, "clock" = %true : i1, !arc.sim.instance<@shiftreg>
      arc.sim.step %model : !arc.sim.instance<@shiftreg>
      arc.sim.set_time %model, %t2 : !arc.sim.instance<@shiftreg>
      arc.sim.set_input %model, "clock" = %false : i1, !arc.sim.instance<@shiftreg>
      arc.sim.step %model : !arc.sim.instance<@shiftreg>

      %res1 = arc.sim.get_port %model, "dout" : i8, !arc.sim.instance<@shiftreg>
      arc.sim.emit "output", %res1 : i8

      arc.sim.set_input %model, "en" = %true : i1, !arc.sim.instance<@shiftreg>

      arc.sim.set_time %model, %t3 : !arc.sim.instance<@shiftreg>
      arc.sim.set_input %model, "clock" = %true : i1, !arc.sim.instance<@shiftreg>
      arc.sim.step %model : !arc.sim.instance<@shiftreg>
      arc.sim.set_time %model, %t4 : !arc.sim.instance<@shiftreg>
      arc.sim.set_input %model, "clock" = %false : i1, !arc.sim.instance<@shiftreg>
      arc.sim.step %model : !arc.sim.instance<@shiftreg>
      %res2 = arc.sim.get_port %model, "dout" : i8, !arc.sim.instance<@shiftreg>
      arc.sim.emit "output", %res2 : i8

      arc.sim.set_time %model, %t5 : !arc.sim.instance<@shiftreg>
      arc.sim.set_input %model, "clock" = %true : i1, !arc.sim.instance<@shiftreg>
      arc.sim.step %model : !arc.sim.instance<@shiftreg>
      arc.sim.set_time %model, %t6 : !arc.sim.instance<@shiftreg>
      arc.sim.set_input %model, "clock" = %false : i1, !arc.sim.instance<@shiftreg>
      arc.sim.step %model : !arc.sim.instance<@shiftreg>
      %res3 = arc.sim.get_port %model, "dout" : i8, !arc.sim.instance<@shiftreg>
      arc.sim.emit "output", %res3 : i8

      arc.sim.set_time %model, %t7 : !arc.sim.instance<@shiftreg>
      arc.sim.set_input %model, "clock" = %true : i1, !arc.sim.instance<@shiftreg>
      arc.sim.step %model : !arc.sim.instance<@shiftreg>
      arc.sim.set_time %model, %t8 : !arc.sim.instance<@shiftreg>
      arc.sim.set_input %model, "clock" = %false : i1, !arc.sim.instance<@shiftreg>
      arc.sim.step %model : !arc.sim.instance<@shiftreg>
      %res4 = arc.sim.get_port %model, "dout" : i8, !arc.sim.instance<@shiftreg>
      arc.sim.emit "output", %res4 : i8

      arc.sim.set_time %model, %t9 : !arc.sim.instance<@shiftreg>
    }
    return
  }
}

// VCD-LABEL: $version
// VCD-NEXT:     Some cryptic ArcRuntime magic
// VCD-NEXT: $end
// VCD-NEXT: $timescale 1fs $end
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
// VCD-NEXT: b11001010 %
// VCD-NEXT: b00000000 &
// VCD-NEXT: b00000000 '
// VCD-NEXT: b11111110 (
// VCD-NEXT: #10
// VCD-NEXT: b11001010 '
// VCD-NEXT: #20
// VCD-NEXT: 1!
// VCD-NEXT: b11111111 $
// VCD-NEXT: #30
// VCD-NEXT: 0!
// VCD-NEXT: #40
// VCD-NEXT: 1!
// VCD-NEXT: 1#
// VCD-NEXT: b00000000 %
// VCD-NEXT: b00000000 '
// VCD-NEXT: b11111111 (
// VCD-NEXT: b11111110 &
// VCD-NEXT: #50
// VCD-NEXT: 0!
// VCD-NEXT: #60
// VCD-NEXT: 1!
// VCD-NEXT: b11111110 %
// VCD-NEXT: b11111110 '
// VCD-NEXT: b11111111 &
// VCD-NEXT: #70
// VCD-NEXT: 0!
// VCD-NEXT: #80
// VCD-NEXT: 1!
// VCD-NEXT: b11111111 %
// VCD-NEXT: b11111111 '
// VCD-NEXT: #90
// VCD-NEXT: 0!
// VCD-NEXT: #100
