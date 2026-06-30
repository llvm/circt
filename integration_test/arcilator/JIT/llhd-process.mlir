// RUN: arcilator --run %s --jit-entry=Foo_main --jit-vcd-file=%t
// RUN: FileCheck %s --check-prefix=VCD --input-file=%t
// REQUIRES: arcilator-jit

// End-to-end test for the `arc-lower-processes` pass. The process yields
// `1`, waits 42ns, yields the value of `%a`, waits 11ns, then halts with
// `2`. `%a` is produced by a call to a trivial function that returns a fixed
// constant, so the auto-generated driver can drive the simulation without
// any external input.

func.func @get_a() -> i32 {
  %c = hw.constant 99 : i32
  return %c : i32
}

hw.module @Foo(out x: i32) {
  %a = func.call @get_a() : () -> i32
  %c1_i32 = hw.constant 1 : i32
  %c2_i32 = hw.constant 2 : i32
  %0 = llhd.constant_time <42000000fs, 0d, 0e>
  %1 = llhd.constant_time <11000000fs, 0d, 0e>
  %2 = llhd.process -> i32 {
    llhd.wait yield (%c1_i32 : i32), delay %0, ^bb1
  ^bb1:
    llhd.wait yield (%a : i32), delay %1, ^bb2
  ^bb2:
    llhd.halt %c2_i32 : i32
  }
  hw.output %2 : i32
}

// VCD:      $scope module Foo $end
// VCD:       $var wire 32 [[X_ID:[^ ]+]] x $end
// VCD:      #0
// VCD:       b{{0*}}1 [[X_ID]]
// VCD:      #42000000
// VCD:       b{{0*}}1100011 [[X_ID]]
// VCD:      #53000000
// VCD:       b{{0*}}10 [[X_ID]]
