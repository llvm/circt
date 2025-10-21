// RUN: arcilator %s --run --jit-entry=main | FileCheck %s
// REQUIRES: arcilator-jit
// CHECK: divu = 00{{$}}
// CHECK: divs = 00{{$}}
// CHECK: modu = 00{{$}}
// CHECK: mods = 00{{$}}

hw.module @Baz(in %a: i8, in %b: i8, out divu: i8, out divs: i8, out modu: i8, out mods: i8) {
  %zero = hw.constant 0 : i8
  %is_zero = comb.icmp eq %zero, %b : i8

  %0 = comb.divu %a, %b : i8
  %1 = comb.mux %is_zero, %zero, %0 : i8

  %2 = comb.divs %a, %b : i8
  %3 = comb.mux %is_zero, %zero, %2 : i8

  %4 = comb.modu %a, %b : i8
  %5 = comb.mux %is_zero, %zero, %4 : i8

  %6 = comb.mods %a, %b : i8
  %7 = comb.mux %is_zero, %zero, %6 : i8

  hw.output %1, %3, %5, %7 : i8, i8, i8, i8
}

func.func @main() {
  %eight = arith.constant 8 : i8
  %zero = arith.constant 0 : i8

  arc.sim.instantiate @Baz as %model {
    arc.sim.set_input %model, "a" = %eight : i8, !arc.sim.instance<@Baz>
    arc.sim.set_input %model, "b" = %zero : i8, !arc.sim.instance<@Baz>
    
    arc.sim.step %model : !arc.sim.instance<@Baz>
  
    %divu = arc.sim.get_port %model, "divu" : i8, !arc.sim.instance<@Baz>
    %divs = arc.sim.get_port %model, "divs" : i8, !arc.sim.instance<@Baz>
    %modu = arc.sim.get_port %model, "modu" : i8, !arc.sim.instance<@Baz>
    %mods = arc.sim.get_port %model, "mods" : i8, !arc.sim.instance<@Baz>

    arc.sim.emit "divu", %divu : i8
    arc.sim.emit "divs", %divs : i8
    arc.sim.emit "modu", %modu : i8
    arc.sim.emit "mods", %mods : i8
  }

  return
}
