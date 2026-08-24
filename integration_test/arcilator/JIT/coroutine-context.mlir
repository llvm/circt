// RUN: arcilator --run %s | FileCheck %s
// REQUIRES: arcilator-jit

// Check that the inferred context is identical at various places.

// CHECK-LABEL: Inner
// CHECK-SAME:  = [[PTR:.+]]{{$}}
// CHECK-NEXT: Outer = [[PTR]]
// CHECK-NEXT: Module = [[PTR]]
// CHECK-NEXT: Instance = [[PTR]]

arc.coroutine.define @Inner() -> () {
  %ctxt = arc.inferred_context
  cf.br ^loop
  ^loop:
  %ptr  = builtin.unrealized_conversion_cast %ctxt: !arc.context to !llvm.ptr
  %int  = llvm.ptrtoint %ptr : !llvm.ptr to i64
  %pre = sim.fmt.literal "Inner = "
  %post = sim.fmt.literal "\n"
  %hex = sim.fmt.hex %int, isUpper false : i64
  %cat = sim.fmt.concat (%pre, %hex, %post)
  sim.proc.print %cat
  arc.coroutine.yield ^loop
}

arc.coroutine.define @Outer(%modCtxt: i64) -> (i1, i64) {
  %state0 = arc.coroutine.undefined_state : !arc.coroutine_state<@Inner>
  %pc0 = arc.coroutine.start_pc : !arc.coroutine_pc<@Inner>
  %0:2 = arc.coroutine.call @Inner(%state0, %pc0) : (!arc.coroutine_state<@Inner>, !arc.coroutine_pc<@Inner>) -> (!arc.coroutine_state<@Inner>, !arc.coroutine_pc<@Inner>)
  cf.br ^loop(%modCtxt : i64)
  ^loop(%arg0: i64) :
  %ctxt = arc.inferred_context
  %ptr  = builtin.unrealized_conversion_cast %ctxt: !arc.context to !llvm.ptr
  %int  = llvm.ptrtoint %ptr : !llvm.ptr to i64
  %preOut = sim.fmt.literal "Outer = "
  %preMod = sim.fmt.literal "Module = "
  %post = sim.fmt.literal "\n"
  %hexOut = sim.fmt.hex %int, isUpper false : i64
  %catOut = sim.fmt.concat (%preOut, %hexOut, %post)
  sim.proc.print %catOut
  %hexMod = sim.fmt.hex %arg0, isUpper false : i64
  %catMod = sim.fmt.concat (%preMod, %hexMod, %post)
  sim.proc.print %catMod
  %mask = arith.constant 0 : i1
  %resume = arith.constant -1 : i64
  arc.coroutine.yield(%mask, %resume: i1, i64),  ^loop
}

hw.module @DUT() {
  %ctxt = arc.inferred_context
  %ptr  = builtin.unrealized_conversion_cast %ctxt: !arc.context to !llvm.ptr
  %int  = llvm.ptrtoint %ptr : !llvm.ptr to i64
  arc.coroutine.instance @Outer(%int) sensitive [false] : (i64) -> ()
}

func.func @entry() -> () {
  arc.sim.instantiate @DUT as %dut {
    arc.sim.step %dut : !arc.sim.instance<@DUT>
    %ctxt = arc.inferred_context
    %ptr  = builtin.unrealized_conversion_cast %ctxt: !arc.context to !llvm.ptr
    %int  = llvm.ptrtoint %ptr : !llvm.ptr to i64
    %pre = sim.fmt.literal "Instance = "
    %post = sim.fmt.literal "\n"
    %hex = sim.fmt.hex %int, isUpper false : i64
    %cat = sim.fmt.concat (%pre, %hex, %post)
    sim.proc.print %cat
  }
  return
}
