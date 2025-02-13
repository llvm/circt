// RUN: arcilator %s --run --jit-entry=main | FileCheck %s
// REQUIRES: arcilator-jit

// CHECK-LABEL: out = cd
// CHECK-NEXT: ARCENV-WARNING: Out-of-bounds array access caught: Index = 6, Size = 6
// CHECK-NEXT: out = ab
// CHECK-NEXT: ARCENV-WARNING: Out-of-bounds array access caught: Index = 7, Size = 6
// CHECK-NEXT: output = ef

module {

  hw.module @array(in %idx: i3, in %front: i8, in %back: i8, out o: i8) {
    %cstCA = hw.constant 0xCA : i8
    %array6xi8 = hw.array_create %back, %cstCA, %cstCA, %cstCA, %cstCA, %front : i8
    %get = hw.array_get %array6xi8[%idx] : !hw.array<6xi8>, i3
    hw.output %get : i8
  }

  func.func @main() {
    %cst5 = arith.constant 5 : i3
    %cst6 = arith.constant 6 : i3
    %cst7 = arith.constant 7 : i3
    %cstAB = arith.constant 0xAB : i8
    %cstCD = arith.constant 0xCD : i8
    %cstEF = arith.constant 0xEF : i8

    arc.sim.instantiate @array as %model {
      arc.sim.set_input %model, "idx" = %cst5 : i3, !arc.sim.instance<@array>
      arc.sim.set_input %model, "front" = %cstAB : i8, !arc.sim.instance<@array>
      arc.sim.set_input %model, "back" = %cstCD : i8, !arc.sim.instance<@array>
      arc.sim.step %model : !arc.sim.instance<@array>

      %0 = arc.sim.get_port %model, "o" : i8, !arc.sim.instance<@array>
      arc.sim.emit "out", %0 : i8

      arc.sim.set_input %model, "idx" = %cst6 : i3, !arc.sim.instance<@array>
      arc.sim.step %model : !arc.sim.instance<@array>

      %1 = arc.sim.get_port %model, "o" : i8, !arc.sim.instance<@array>
      arc.sim.emit "out", %1 : i8

      arc.sim.set_input %model, "idx" = %cst7 : i3, !arc.sim.instance<@array>
      arc.sim.set_input %model, "front" = %cstEF : i8, !arc.sim.instance<@array>
      arc.sim.step %model : !arc.sim.instance<@array>

      %2 = arc.sim.get_port %model, "o" : i8, !arc.sim.instance<@array>
      arc.sim.emit "out", %2 : i8
    }
    return
  }
}
