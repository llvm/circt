// RUN: arcilator %s --run --jit-entry=main | FileCheck %s
// REQUIRES: arcilator-jit

// COM: Verify roundtrip and field extraction for non-power-of-two (odd-width)
// COM: struct members (!hw.struct<f3: i3, f5: i5, f7: i7>, total 15 bits).
// CHECK: match = 1

hw.module @oddStructMod(in %clk: i1, in %in_val: i15, out out_val: i15, out match: i1) {
  %seq_clk = seq.to_clock %clk
  %struct = hw.bitcast %in_val : (i15) -> !hw.struct<f3: i3, f5: i5, f7: i7>
  %reg = seq.compreg %struct, %seq_clk : !hw.struct<f3: i3, f5: i5, f7: i7>
  %in_reg = seq.compreg %in_val, %seq_clk : i15

  %f3 = hw.struct_extract %reg["f3"] : !hw.struct<f3: i3, f5: i5, f7: i7>
  %f5 = hw.struct_extract %reg["f5"] : !hw.struct<f3: i3, f5: i5, f7: i7>
  %f7 = hw.struct_extract %reg["f7"] : !hw.struct<f3: i3, f5: i5, f7: i7>

  %reconstructed = hw.struct_create(%f3, %f5, %f7) : !hw.struct<f3: i3, f5: i5, f7: i7>
  %out_val = hw.bitcast %reconstructed : (!hw.struct<f3: i3, f5: i5, f7: i7>) -> i15

  %match = comb.icmp eq %in_reg, %out_val : i15
  hw.output %out_val, %match : i15, i1
}

func.func @main() {
  %zero = arith.constant 0 : i1
  %one = arith.constant 1 : i1
  %cst = arith.constant 27397 : i15 // 0x6B05 binary: 110_10110_0000101

  arc.sim.instantiate @oddStructMod as %model {
    arc.sim.set_input %model, "in_val" = %cst : i15, !arc.sim.instance<@oddStructMod>

    arc.sim.set_input %model, "clk" = %one : i1, !arc.sim.instance<@oddStructMod>
    arc.sim.step %model : !arc.sim.instance<@oddStructMod>
    arc.sim.set_input %model, "clk" = %zero : i1, !arc.sim.instance<@oddStructMod>
    arc.sim.step %model : !arc.sim.instance<@oddStructMod>

    arc.sim.set_input %model, "clk" = %one : i1, !arc.sim.instance<@oddStructMod>
    arc.sim.step %model : !arc.sim.instance<@oddStructMod>
    arc.sim.set_input %model, "clk" = %zero : i1, !arc.sim.instance<@oddStructMod>
    arc.sim.step %model : !arc.sim.instance<@oddStructMod>

    %match = arc.sim.get_port %model, "match" : i1, !arc.sim.instance<@oddStructMod>
    arc.sim.emit "match", %match : i1
  }

  return
}

