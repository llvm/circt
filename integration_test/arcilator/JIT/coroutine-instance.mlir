// RUN: arcilator --run %s --jit-entry=entry --jit-vcd-file=%t | FileCheck %s
// RUN: FileCheck %s --check-prefix=VCD --input-file=%t
// REQUIRES: arcilator-jit

// End-to-end test for the `arc.coroutine.instance` lowering through the
// arcilator JIT, mimicking what an `llhd.process` should eventually lower to.
//
// A coroutine cannot read the model storage directly, so the module reads the
// current simulation time with `llhd.current_time` and passes it into `@Proc`
// as an argument on every entry. The coroutine schedules its next wakeup as
// `now + delay`.

// CHECK:      A 0 @ 0
// CHECK-NEXT: out=42
// CHECK-NEXT: A 1 @ 100
// CHECK-NEXT: out=43
// CHECK-NEXT: A 2 @ 200
// CHECK-NEXT: out=44
// CHECK-NEXT: A 3 @ 300
// CHECK-NEXT: out=45
// CHECK-NEXT: A 4 @ 400
// CHECK-NEXT: out=46
// CHECK-NEXT: B 0 @ 500
// CHECK-NEXT: out=42
// CHECK-NEXT: B 1 @ 511
// CHECK-NEXT: out=43
// CHECK-NEXT: B 2 @ 522
// CHECK-NEXT: out=44
// CHECK-NEXT: B 3 @ 533
// CHECK-NEXT: out=45
// CHECK-NEXT: B 4 @ 544
// CHECK-NEXT: out=46
// CHECK-NEXT: out=47

// Two counting loops with different suspension delays. The current time is
// supplied as the `%now` argument on every entry. Each iteration yields the
// loop counter as the `i42` result and `now + delay` as the final `i64` result
// the instance uses to schedule the next wakeup.
arc.coroutine.define @Proc(%now: i64) -> (i42, i64) {
  // Loop A: count to 5, suspending 100 time units each step.
  %c0 = hw.constant 0 : i42
  cf.br ^loopA(%now, %c0 : i64, i42)

^loopA(%tA: i64, %cA: i42):
  %five = hw.constant 5 : i42
  %goA = comb.icmp ult %cA, %five : i42
  cf.cond_br %goA, ^bodyA(%tA, %cA : i64, i42), ^initB(%tA : i64)

^bodyA(%t: i64, %c: i42):
  %litA = sim.fmt.literal "A "
  %decA = sim.fmt.dec %c specifierWidth 0 : i42
  %at = sim.fmt.literal " @ "
  %timeA = sim.fmt.dec %t specifierWidth 0 : i64
  %nlA = sim.fmt.literal "\n"
  %msgA = sim.fmt.concat (%litA, %decA, %at, %timeA, %nlA)
  sim.proc.print %msgA
  %d100 = hw.constant 100 : i64
  %wakeA = comb.add %t, %d100 : i64
  %one = hw.constant 1 : i42
  %cnA = comb.add %c, %one : i42
  arc.coroutine.yield (%c, %wakeA : i42, i64), ^resumeA(%cnA : i42)

^resumeA(%nowA: i64, %rcA: i42):
  cf.br ^loopA(%nowA, %rcA : i64, i42)

  // Loop B: count to 5, suspending 11 time units each step.
^initB(%tB: i64):
  %d0 = hw.constant 0 : i42
  cf.br ^loopB(%tB, %d0 : i64, i42)

^loopB(%tLB: i64, %dLB: i42):
  %five2 = hw.constant 5 : i42
  %goB = comb.icmp ult %dLB, %five2 : i42
  cf.cond_br %goB, ^bodyB(%tLB, %dLB : i64, i42), ^done

^bodyB(%t2: i64, %d2: i42):
  %litB = sim.fmt.literal "B "
  %decB = sim.fmt.dec %d2 specifierWidth 0 : i42
  %at2 = sim.fmt.literal " @ "
  %timeB = sim.fmt.dec %t2 specifierWidth 0 : i64
  %nlB = sim.fmt.literal "\n"
  %msgB = sim.fmt.concat (%litB, %decB, %at2, %timeB, %nlB)
  sim.proc.print %msgB
  %d11 = hw.constant 11 : i64
  %wakeB = comb.add %t2, %d11 : i64
  %one2 = hw.constant 1 : i42
  %dnB = comb.add %d2, %one2 : i42
  arc.coroutine.yield (%d2, %wakeB : i42, i64), ^resumeB(%dnB : i42)

^resumeB(%nowB: i64, %rdB: i42):
  cf.br ^loopB(%nowB, %rdB : i64, i42)

  // Done: latch the final count and halt forever.
^done:
  %final = hw.constant 5 : i42
  %never = hw.constant -1 : i64
  arc.coroutine.halt %final, %never : i42, i64
}

// Reads the current simulation time, runs `@Proc` with it, and exposes a
// derived output: the coroutine count plus 42.
hw.module @CoroutineProc(out o: i42) {
  %now = llhd.current_time
  %nowi = llhd.time_to_int %now
  %0 = arc.coroutine.instance @Proc(%nowi) : (i64) -> i42
  %c42 = hw.constant 42 : i42
  %sum = comb.add %0, %c42 : i42
  hw.output %sum : i42
}

// Drives the model like a simulator: query the next wakeup, advance time to it,
// evaluate, and print the latched output. Stops once no wakeup is pending.
func.func @entry() {
  arc.sim.instantiate @CoroutineProc as %inst {
    %never = hw.constant -1 : i64
    %w0 = arc.sim.get_next_wakeup %inst : !arc.sim.instance<@CoroutineProc>

    scf.while (%w = %w0) : (i64) -> i64 {
      %pending = comb.icmp ne %w, %never : i64
      scf.condition(%pending) %w : i64
    } do {
    ^bb0(%w: i64):
      arc.sim.set_time %inst, %w : !arc.sim.instance<@CoroutineProc>
      arc.sim.step %inst : !arc.sim.instance<@CoroutineProc>

      %out = arc.sim.get_port %inst, "o" : i42, !arc.sim.instance<@CoroutineProc>
      %lit = sim.fmt.literal "out="
      %dec = sim.fmt.dec %out specifierWidth 0 : i42
      %nl = sim.fmt.literal "\n"
      %msg = sim.fmt.concat (%lit, %dec, %nl)
      sim.proc.print %msg

      %wn = arc.sim.get_next_wakeup %inst : !arc.sim.instance<@CoroutineProc>
      scf.yield %wn : i64
    }
  }
  return
}

// The VCD trace records `o` against simulation time, advancing in steps of 100
// for loop A and 11 for loop B. Leading zeros of the 42-bit signal are skipped
// with `{{0*}}`, and the signal identifier is captured as `[[ID]]`.

// VCD:       $scope module CoroutineProc $end
// VCD-NEXT:   $var wire 42 [[ID:[^ ]+]] o $end
// VCD:       #0
// VCD-NEXT:  b{{0*}} [[ID]]
// VCD-NEXT:  #0
// VCD-NEXT:  b{{0*}}101010 [[ID]]
// VCD-NEXT:  #100
// VCD-NEXT:  b{{0*}}101011 [[ID]]
// VCD-NEXT:  #200
// VCD-NEXT:  b{{0*}}101100 [[ID]]
// VCD-NEXT:  #300
// VCD-NEXT:  b{{0*}}101101 [[ID]]
// VCD-NEXT:  #400
// VCD-NEXT:  b{{0*}}101110 [[ID]]
// VCD-NEXT:  #500
// VCD-NEXT:  b{{0*}}101010 [[ID]]
// VCD-NEXT:  #511
// VCD-NEXT:  b{{0*}}101011 [[ID]]
// VCD-NEXT:  #522
// VCD-NEXT:  b{{0*}}101100 [[ID]]
// VCD-NEXT:  #533
// VCD-NEXT:  b{{0*}}101101 [[ID]]
// VCD-NEXT:  #544
// VCD-NEXT:  b{{0*}}101110 [[ID]]
// VCD-NEXT:  #555
// VCD-NEXT:  b{{0*}}101111 [[ID]]
