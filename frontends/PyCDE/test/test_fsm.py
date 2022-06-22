# RUN: %PYTHON% py-split-input-file.py %s | FileCheck %s

from re import A
from pycde import System, Input, Output, module, generator, externmodule

from pycde.dialects import comb
from pycde import fsm
from pycde.pycde_types import types, dim

from circt.support import connect

# FSM instantiation example

# CHECK-LABEL: msft.module @FSMUser {} (%a: i1, %b: i1, %clk: i1) -> (is_a: i1, is_b: i1) attributes {fileName = "FSMUser.sv"} {
# CHECK:         %FSM.is_A, %FSM.is_B = msft.instance @FSM @FSM(%a, %b, %clk)  : (i1, i1, i1) -> (i1, i1)
# CHECK:         msft.output %FSM.is_A, %FSM.is_B : i1, i1
# CHECK:       }
# CHECK-LABEL: msft.module @FSM {} (%a: i1, %b: i1, %clock: i1) -> (is_A: i1, is_B: i1) attributes {fileName = "FSM.sv"} {
# CHECK:         %0:2 = "fsm.hw_instance"(%a, %b, %clock) {machine = "FSM_impl", operand_segment_sizes = dense<[2, 1, 0]> : vector<3xi32>, sym_name = "FSM"} : (i1, i1, i1) -> (i1, i1)
# CHECK:         msft.output %0#0, %0#1 : i1, i1
# CHECK:       }


@fsm.machine(clock="clock")
class FSM:
  a = Input(types.i1)
  b = Input(types.i1)

  fsm_initial_state = 'A'
  fsm_transitions = {
      'A': ('B', lambda ports: ports.a),
      'B': ('A', lambda ports: ports.b),
  }


@module
class FSMUser:
  a = Input(types.i1)
  b = Input(types.i1)
  clk = Input(types.i1)
  is_a = Output(types.i1)
  is_b = Output(types.i1)

  @generator
  def construct(ports):
    fsm = FSM(a=ports.a, b=ports.b, clock=ports.clk)
    ports.is_a = fsm.is_A
    ports.is_b = fsm.is_B


system = System([FSMUser])
system.generate()
system.print()

# -----

# FSM state transitions example

# CHECK: "fsm.machine"() ({
# CHECK:   ^bb0(%arg0: i1, %arg1: i1, %arg2: i1):
# CHECK:     %false = hw.constant false
# CHECK:     %true = hw.constant true
# CHECK:     "fsm.state"() ({
# CHECK:       "fsm.output"(%true, %false, %false, %false) : (i1, i1, i1, i1) -> ()
# CHECK:     }, {
# CHECK:       "fsm.transition"() ({
# CHECK:         "fsm.return"(%arg0) : (i1) -> ()
# CHECK:       }, {
# CHECK:       }) {nextState = @a} : () -> ()
# CHECK:     }) {sym_name = "idle"} : () -> ()
# CHECK:     "fsm.state"() ({
# CHECK:       "fsm.output"(%false, %true, %false, %false) : (i1, i1, i1, i1) -> ()
# CHECK:     }, {
# CHECK:       "fsm.transition"() ({
# CHECK:       }, {
# CHECK:       }) {nextState = @b} : () -> ()
# CHECK:     }) {sym_name = "a"} : () -> ()
# CHECK:     "fsm.state"() ({
# CHECK:       "fsm.output"(%false, %false, %true, %false) : (i1, i1, i1, i1) -> ()
# CHECK:     }, {
# CHECK:       "fsm.transition"() ({
# CHECK:         %0 = comb.and %arg0, %arg1 : i1
# CHECK:         %true_0 = hw.constant true
# CHECK:         %1 = comb.xor %0, %true_0 : i1
# CHECK:         %2 = comb.and %arg1, %arg2 : i1
# CHECK:         %true_1 = hw.constant true
# CHECK:         %3 = comb.xor %2, %true_1 : i1
# CHECK:         %4 = comb.and %arg0, %arg2 : i1
# CHECK:         %true_2 = hw.constant true
# CHECK:         %5 = comb.xor %4, %true_2 : i1
# CHECK:         %6 = comb.and %1, %3, %5 : i1
# CHECK:         %true_3 = hw.constant true
# CHECK:         %7 = comb.xor %6, %true_3 : i1
# CHECK:         "fsm.return"(%7) : (i1) -> ()
# CHECK:       }, {
# CHECK:       }) {nextState = @c} : () -> ()
# CHECK:     }) {sym_name = "b"} : () -> ()
# CHECK:     "fsm.state"() ({
# CHECK:       "fsm.output"(%false, %false, %false, %true) : (i1, i1, i1, i1) -> ()
# CHECK:     }, {
# CHECK:       "fsm.transition"() ({
# CHECK:         "fsm.return"(%arg2) : (i1) -> ()
# CHECK:       }, {
# CHECK:       }) {nextState = @idle} : () -> ()
# CHECK:       "fsm.transition"() ({
# CHECK:         %true_0 = hw.constant true
# CHECK:         %0 = comb.xor %arg1, %true_0 : i1
# CHECK:         "fsm.return"(%0) : (i1) -> ()
# CHECK:       }, {
# CHECK:       }) {nextState = @a} : () -> ()
# CHECK:     }) {sym_name = "c"} : () -> ()
# CHECK:   }) {clock_name = "clock", function_type = (i1, i1, i1) -> (i1, i1, i1, i1), in_names = ["a", "b", "c"], initialState = "idle", out_names = ["is_a", "is_b", "is_c", "is_idle"], sym_name = "F0_impl"} : () -> ()


@fsm.machine(clock="clock")
class F0:
  a = Input(types.i1)
  b = Input(types.i1)
  c = Input(types.i1)

  def maj3(ports):

    def nand(*args):
      return comb.XorOp(comb.AndOp(*args), types.i1(1))

    c1 = nand(ports.a, ports.b)
    c2 = nand(ports.b, ports.c)
    c3 = nand(ports.a, ports.c)
    return nand(c1, c2, c3)

  fsm_initial_state = 'idle'
  fsm_transitions = {
      # Always taken transition
      'idle':
          'a',
      # Transition using inline function
      'a': ('b', lambda ports: ports.a),
      # Transition using outlined function
      'b': ('c', maj3),
      # Multiple transitions
      'c': [('idle', lambda ports: ports.c),
            ('a', lambda ports: comb.XorOp(ports.b, types.i1(1)))],
  }


system = System([F0])
system.generate()
system.print()
