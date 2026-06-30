// RUN: arcilator --run %s --jit-entry=entry | FileCheck %s
// REQUIRES: arcilator-jit

// This test exercises the Arc coroutine lowering end-to-end through the
// arcilator JIT. It defines two coroutines to cover nesting:
//
// - `@child` is a leaf coroutine that suspends twice and then returns. It
//   carries an SSA value across each suspension point to exercise state
//   persistence.
//
// - `@parent` drives `@child` to completion, accumulating the yielded values.
//   Between child steps it suspends itself, carrying the child's coroutine
//   state and program counter across its own suspension points. This embeds a
//   nested coroutine's state inside the parent's persisted state.
//
// The `@entry` function then drives `@parent` from an `scf.while` loop,
// re-entering it until it returns or a timeout is hit, printing every value it
// receives. Coroutines print as they make progress so the interleaving is
// visible to FileCheck.

// CHECK:      [child] enter, yield 100
// CHECK-NEXT: [parent] yield running sum = 100
// CHECK-NEXT: [entry] received = 100
// CHECK-NEXT: [child] resume, yield 200
// CHECK-NEXT: [parent] yield running sum = 300
// CHECK-NEXT: [entry] received = 300
// CHECK-NEXT: [child] resume, return 300
// CHECK-NEXT: [parent] return total = 600
// CHECK-NEXT: [entry] received = 600

// A leaf coroutine that yields 100, then 200, then returns 300. The value
// produced before each yield is carried across the suspension point as the
// resume block's persisted argument.
arc.coroutine.define @child() -> i32 {
  %lit = sim.fmt.literal "[child] enter, yield 100\n"
  sim.proc.print %lit
  %v1 = hw.constant 100 : i32
  arc.coroutine.yield (%v1 : i32), ^resume1(%v1 : i32)
^resume1(%prev1: i32):
  %lit1 = sim.fmt.literal "[child] resume, yield 200\n"
  sim.proc.print %lit1
  %c100 = hw.constant 100 : i32
  %v2 = comb.add %prev1, %c100 : i32
  arc.coroutine.yield (%v2 : i32), ^resume2(%v2 : i32)
^resume2(%prev2: i32):
  %lit2 = sim.fmt.literal "[child] resume, return 300\n"
  sim.proc.print %lit2
  %c100b = hw.constant 100 : i32
  %v3 = comb.add %prev2, %c100b : i32
  arc.coroutine.return %v3 : i32
}

// Drives `@child` in a loop, summing the yielded values. The child's state
// and program counter, along with the running sum, are carried across the
// parent's own suspension points.
arc.coroutine.define @parent() -> i32 {
  %state0 = arc.coroutine.undefined_state : !arc.coroutine_state<@child>
  %pc0 = arc.coroutine.start_pc : !arc.coroutine_pc<@child>
  %sum0 = hw.constant 0 : i32
  cf.br ^loop(%state0, %pc0, %sum0 : !arc.coroutine_state<@child>, !arc.coroutine_pc<@child>, i32)

^loop(%cs: !arc.coroutine_state<@child>, %cp: !arc.coroutine_pc<@child>, %sum: i32):
  %ns, %np, %r = arc.coroutine.call @child(%cs, %cp) : (!arc.coroutine_state<@child>, !arc.coroutine_pc<@child>) -> (!arc.coroutine_state<@child>, !arc.coroutine_pc<@child>, i32)
  %nsum = comb.add %sum, %r : i32
  %done = arc.coroutine.pc_is_return %np : !arc.coroutine_pc<@child>
  cf.cond_br %done, ^finish(%nsum : i32), ^suspend(%ns, %np, %nsum : !arc.coroutine_state<@child>, !arc.coroutine_pc<@child>, i32)

^suspend(%ss: !arc.coroutine_state<@child>, %sp: !arc.coroutine_pc<@child>, %ssum: i32):
  %lit = sim.fmt.literal "[parent] yield running sum = "
  %dec = sim.fmt.dec %ssum : i32
  %nl = sim.fmt.literal "\n"
  %msg = sim.fmt.concat (%lit, %dec, %nl)
  sim.proc.print %msg
  arc.coroutine.yield (%ssum : i32), ^resume(%ss, %sp, %ssum : !arc.coroutine_state<@child>, !arc.coroutine_pc<@child>, i32)

^resume(%rs: !arc.coroutine_state<@child>, %rp: !arc.coroutine_pc<@child>, %rsum: i32):
  cf.br ^loop(%rs, %rp, %rsum : !arc.coroutine_state<@child>, !arc.coroutine_pc<@child>, i32)

^finish(%total: i32):
  %litf = sim.fmt.literal "[parent] return total = "
  %decf = sim.fmt.dec %total : i32
  %nlf = sim.fmt.literal "\n"
  %msgf = sim.fmt.concat (%litf, %decf, %nlf)
  sim.proc.print %msgf
  arc.coroutine.return %total : i32
}

// Repeatedly re-enters `@parent` until it returns or a timeout is reached,
// printing every value the coroutine produces.
func.func @entry() {
  %state0 = arc.coroutine.undefined_state : !arc.coroutine_state<@parent>
  %pc0 = arc.coroutine.start_pc : !arc.coroutine_pc<@parent>
  %iter0 = hw.constant 0 : i32
  %limit = hw.constant 100 : i32
  %true = hw.constant true

  scf.while (%s = %state0, %p = %pc0, %i = %iter0) : (!arc.coroutine_state<@parent>, !arc.coroutine_pc<@parent>, i32) -> (!arc.coroutine_state<@parent>, !arc.coroutine_pc<@parent>, i32) {
    // Keep going until the coroutine returns, but never longer than the
    // timeout to guard against a runaway loop.
    %done = arc.coroutine.pc_is_return %p : !arc.coroutine_pc<@parent>
    %notDone = comb.xor %done, %true : i1
    %withinTimeout = comb.icmp ult %i, %limit : i32
    %cond = comb.and %notDone, %withinTimeout : i1
    scf.condition(%cond) %s, %p, %i : !arc.coroutine_state<@parent>, !arc.coroutine_pc<@parent>, i32
  } do {
  ^bb0(%s: !arc.coroutine_state<@parent>, %p: !arc.coroutine_pc<@parent>, %i: i32):
    %ns, %np, %r = arc.coroutine.call @parent(%s, %p) : (!arc.coroutine_state<@parent>, !arc.coroutine_pc<@parent>) -> (!arc.coroutine_state<@parent>, !arc.coroutine_pc<@parent>, i32)
    %lit = sim.fmt.literal "[entry] received = "
    %dec = sim.fmt.dec %r : i32
    %nl = sim.fmt.literal "\n"
    %msg = sim.fmt.concat (%lit, %dec, %nl)
    sim.proc.print %msg
    %c1 = hw.constant 1 : i32
    %ni = comb.add %i, %c1 : i32
    scf.yield %ns, %np, %ni : !arc.coroutine_state<@parent>, !arc.coroutine_pc<@parent>, i32
  }

  return
}
