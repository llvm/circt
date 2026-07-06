// RUN: arcilator --run %s --jit-entry=Foo_main | FileCheck %s
// REQUIRES: arcilator-jit

// End-to-end test for observed values on `llhd.wait`. A driver process toggles
// two values `%a` and `%b` on a fixed schedule, and a consumer process waits on
// changes to just `%a`, just `%b`, both, and both-or-a-delay, reporting on each
// wake-up. It exercises two properties of the observe-bitmask lowering:
//
//   - A single-signal wait ignores changes on the other signal. The driver
//     deliberately toggles the *unobserved* signal one step before each real
//     wake-up (a decoy), and those decoys must not resume the consumer.
//
//   - The wake happens at the exact time the observed value changes, with no
//     one-event lag, because coroutine arguments are sampled in the New phase.
//
// Driver schedule (10fs steps); `%a` counts by 2, `%b` by 3:
//
//   t=0    a=0 b=0     initial                 -> report 0
//   t=10   a=0 b=3     b decoy (a-wait armed)  -> ignored
//   t=20   a=2 b=3     a changes               -> report 1
//   t=30   a=4 b=3     a decoy (b-wait armed)  -> ignored
//   t=40   a=4 b=6     b changes               -> report 2
//   t=50   a=6 b=9     a and b change          -> report 3
//   t=53   a=6 b=9     5fs... 3fs delay fires  -> report 4

// CHECK:      0 @0: a=0 b=0
// CHECK-NEXT: 1 @20: a=2 b=3
// CHECK-NEXT: 2 @40: a=4 b=6
// CHECK-NEXT: 3 @50: a=6 b=9
// CHECK-NEXT: 4 @53: a=6 b=9

hw.module @Foo() {
  // Driver process producing a/b values that change over time.
  %a, %b = llhd.process -> i17, i42 {
    %d10 = llhd.constant_time <10fs, 0d, 0e>
    %a0 = hw.constant 0 : i17
    %b0 = hw.constant 0 : i42
    llhd.wait yield (%a0, %b0 : i17, i42), delay %d10, ^step1
  ^step1:
    // t=10: toggle b only (decoy for the wait on a).
    %b1 = hw.constant 3 : i42
    llhd.wait yield (%a0, %b1 : i17, i42), delay %d10, ^step2
  ^step2:
    // t=20: change a.
    %a1 = hw.constant 2 : i17
    llhd.wait yield (%a1, %b1 : i17, i42), delay %d10, ^step3
  ^step3:
    // t=30: toggle a only (decoy for the wait on b).
    %a2 = hw.constant 4 : i17
    llhd.wait yield (%a2, %b1 : i17, i42), delay %d10, ^step4
  ^step4:
    // t=40: change b.
    %b2 = hw.constant 6 : i42
    llhd.wait yield (%a2, %b2 : i17, i42), delay %d10, ^step5
  ^step5:
    // t=50: change a and b, then halt.
    %a3 = hw.constant 6 : i17
    %b3 = hw.constant 9 : i42
    llhd.halt %a3, %b3 : i17, i42
  }

  // Consumer process waiting for changes on a/b.
  %now = llhd.current_time
  %nowi = llhd.time_to_int %now
  llhd.process {
    %i0 = hw.constant 0 : i32
    %i1 = hw.constant 1 : i32
    %i2 = hw.constant 2 : i32
    %i3 = hw.constant 3 : i32
    %i4 = hw.constant 4 : i32
    func.call @report(%i0, %nowi, %a, %b) : (i32, i64, i17, i42) -> ()
    llhd.wait (%a : i17), ^wait_a
  ^wait_a:
    func.call @report(%i1, %nowi, %a, %b) : (i32, i64, i17, i42) -> ()
    llhd.wait (%b : i42), ^wait_b
  ^wait_b:
    func.call @report(%i2, %nowi, %a, %b) : (i32, i64, i17, i42) -> ()
    llhd.wait (%a, %b : i17, i42), ^wait_ab
  ^wait_ab:
    func.call @report(%i3, %nowi, %a, %b) : (i32, i64, i17, i42) -> ()
    %d3 = llhd.constant_time <3fs, 0d, 0e>
    llhd.wait delay %d3, (%a, %b : i17, i42), ^wait_delay
  ^wait_delay:
    func.call @report(%i4, %nowi, %a, %b) : (i32, i64, i17, i42) -> ()
    llhd.halt
  }
}

func.func @report(%idx: i32, %time: i64, %a: i17, %b: i42) {
  %0 = sim.fmt.dec %idx specifierWidth 0 : i32
  %1 = sim.fmt.literal " @"
  %2 = sim.fmt.dec %time specifierWidth 0 : i64
  %3 = sim.fmt.literal ": a="
  %4 = sim.fmt.dec %a specifierWidth 0 : i17
  %5 = sim.fmt.literal " b="
  %6 = sim.fmt.dec %b specifierWidth 0 : i42
  %7 = sim.fmt.literal "\n"
  %8 = sim.fmt.concat (%0, %1, %2, %3, %4, %5, %6, %7)
  sim.proc.print %8
  return
}
