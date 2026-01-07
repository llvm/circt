// RUN: arcilator %s --run --jit-entry=roundtrip | FileCheck %s --check-prefixes=RTRIP
// RUN: arcilator %s --run --jit-entry=orderTest | FileCheck %s --check-prefixes=ORDER
// REQUIRES: arcilator-jit

// COM: Skip first two cycles
// RTRIP-COUNT-2: match
// RTRIP-COUNT-1024: match = 1

hw.module @bitcaster(in %clk: i1, in %i: i10, out o: i10, out match: i1) {
  %seq_clk = seq.to_clock %clk
  %q = seq.compreg %i, %seq_clk : i10
  %qq = seq.compreg %q, %seq_clk : i10
  %array = hw.bitcast %i : (i10) -> !hw.array<5xi2>
  %arrayReg = seq.compreg %array, %seq_clk : !hw.array<5xi2>
  %struct = hw.bitcast %arrayReg : (!hw.array<5xi2>) -> !hw.struct<a: i1, b: !hw.array<7xi1>, c: i2>
  %structReg = seq.compreg %struct, %seq_clk : !hw.struct<a: i1, b: !hw.array<7xi1>, c: i2>
  %out = hw.bitcast %structReg : (!hw.struct<a: i1, b: !hw.array<7xi1>, c: i2>) -> i10
  %match = comb.icmp eq %qq, %out: i10
  hw.output %out, %match : i10, i1
}

// ORDER:      ae0 = b
// ORDER-NEXT: ae1 = a
// ORDER-NEXT: se0 = a
// ORDER-NEXT: se1 = b

hw.module @arrayMod(in %raw: i8, out e0 : i4, out e1 : i4) {
  %cst0 = hw.constant 0 : i1
  %cst1 = hw.constant 1 : i1
  %array = hw.bitcast %raw : (i8) -> !hw.array<2xi4>
  %e0 = hw.array_get %array[%cst0] : !hw.array<2xi4>, i1
  %e1 = hw.array_get %array[%cst1] : !hw.array<2xi4>, i1
  hw.output %e0, %e1 : i4, i4
}

hw.module @structMod(in %raw: i8, out e0 : i4, out e1 : i4) {
  %struct = hw.bitcast %raw : (i8) -> !hw.struct<a: i4, b: i4>
  %e0 = hw.struct_extract %struct["a"] : !hw.struct<a: i4, b: i4>
  %e1 = hw.struct_extract %struct["b"] :!hw.struct<a: i4, b: i4>
  hw.output %e0, %e1 : i4, i4
}

func.func @roundtrip() {
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
      %o = arc.sim.get_port %model, "o" : i10, !arc.sim.instance<@bitcaster>
      arc.sim.emit "out", %o : i10
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

func.func @orderTest() {
  %cstAB = arith.constant 0xab : i8

  arc.sim.instantiate @arrayMod as %model {
    arc.sim.set_input %model, "raw" = %cstAB : i8, !arc.sim.instance<@arrayMod>
    arc.sim.step %model : !arc.sim.instance<@arrayMod>
    %e0 = arc.sim.get_port %model, "e0" : i4, !arc.sim.instance<@arrayMod>
    %e1 = arc.sim.get_port %model, "e1" : i4, !arc.sim.instance<@arrayMod>
    arc.sim.emit "ae0", %e0 : i4
    arc.sim.emit "ae1", %e1 : i4
  }

  arc.sim.instantiate @structMod as %model {
    arc.sim.set_input %model, "raw" = %cstAB : i8, !arc.sim.instance<@structMod>
    arc.sim.step %model : !arc.sim.instance<@structMod>
    %e0 = arc.sim.get_port %model, "e0" : i4, !arc.sim.instance<@structMod>
    %e1 = arc.sim.get_port %model, "e1" : i4, !arc.sim.instance<@structMod>
    arc.sim.emit "se0", %e0 : i4
    arc.sim.emit "se1", %e1 : i4
  }

  return
}
