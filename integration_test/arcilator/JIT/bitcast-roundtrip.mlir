// RUN: arcilator %s --run --jit-entry=main | FileCheck %s
// REQUIRES: arcilator-jit

// COM: Skip first two cycles
// CHECK-COUNT-2: match
// CHECK-COUNT-1024: match = 1

hw.module @bitcaster(in %clk: i1, in %i: i10, out o: i10, out match: i1) {
  %seq_clk = seq.to_clock %clk
  %q = seq.compreg %i, %seq_clk : i10
  %qq = seq.compreg %q, %seq_clk : i10
  %array = hw.bitcast %i : (i10) -> !hw.array<5xi2>
  %arrayReg = seq.compreg %array, %seq_clk : !hw.array<5xi2>
  %struct = hw.bitcast %arrayReg : (!hw.array<5xi2>) -> !hw.struct<a: i1, b: i7, c: i2>
  %structReg = seq.compreg %struct, %seq_clk : !hw.struct<a: i1, b: i7, c: i2>
  %out = hw.bitcast %structReg : (!hw.struct<a: i1, b: i7, c: i2>) -> i10
  %match = comb.icmp eq %qq, %out: i10
  hw.output %out, %match : i10, i1
}

func.func @main() {
  %zero = arith.constant 0 : i1
  %one = arith.constant 1 : i1
  %lb = arith.constant 0 : index
  %ub = arith.constant 1024 : index
  %step = arith.constant 1 : index

  arc.sim.instantiate @bitcaster as %model {
    scf.for %i = %lb to %ub step %step {
      %raw = arith.index_castui %i : index to i10
      arc.sim.set_input %model, "i" = %raw : i10, !arc.sim.instance<@bitcaster>

      arc.sim.set_input %model, "clk" = %one : i1, !arc.sim.instance<@bitcaster>
      arc.sim.step %model : !arc.sim.instance<@bitcaster>
      arc.sim.set_input %model, "clk" = %zero : i1, !arc.sim.instance<@bitcaster>
      arc.sim.step %model : !arc.sim.instance<@bitcaster>

      %res = arc.sim.get_port %model, "match" : i1, !arc.sim.instance<@bitcaster>
      arc.sim.emit "match", %res : i1
    }

    arc.sim.set_input %model, "clk" = %one : i1, !arc.sim.instance<@bitcaster>
    arc.sim.step %model : !arc.sim.instance<@bitcaster>
    arc.sim.set_input %model, "clk" = %zero : i1, !arc.sim.instance<@bitcaster>
    arc.sim.step %model : !arc.sim.instance<@bitcaster>
    %res0 = arc.sim.get_port %model, "match" : i1, !arc.sim.instance<@bitcaster>
    arc.sim.emit "match", %res0 : i1

    arc.sim.set_input %model, "clk" = %one : i1, !arc.sim.instance<@bitcaster>
    arc.sim.step %model : !arc.sim.instance<@bitcaster>
    arc.sim.set_input %model, "clk" = %zero : i1, !arc.sim.instance<@bitcaster>
    arc.sim.step %model : !arc.sim.instance<@bitcaster>
    %res1 = arc.sim.get_port %model, "match" : i1, !arc.sim.instance<@bitcaster>
    arc.sim.emit "match", %res1 : i1
  }

  return
}