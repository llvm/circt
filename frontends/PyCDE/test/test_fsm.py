# RUN: %PYTHON% py-split-input-file.py %s | FileCheck %s
# XFAIL: *

from pycde import System, Input, Output, generator, Module
from pycde.dialects import comb
from pycde import fsm
from pycde.types import types
from pycde.testing import unittestmodule

# FSM instantiation example

# CHECK-LABEL:  msft.module @FSMUser {} (%a: i1, %b: i1, %c: i1, %clk: i1, %rst: i1) -> (is_a: i1, is_b: i1, is_c: i1) attributes {fileName = "FSMUser.sv"} {
# CHECK:          %FSM.is_A, %FSM.is_B, %FSM.is_C = msft.instance @FSM @FSM(%a, %b, %c, %clk, %rst)  : (i1, i1, i1, i1, i1) -> (i1, i1, i1)
# CHECK:          msft.output %FSM.is_A, %FSM.is_B, %FSM.is_C : i1, i1, i1
# CHECK:        }
# CHECK-LABEL:  msft.module @FSM {} (%a: i1, %b: i1, %c: i1, %clk: i1, %rst: i1) -> (is_A: i1, is_B: i1, is_C: i1) attributes {fileName = "FSM.sv"} {
# CHECK:          %0:3 = fsm.hw_instance "FSM_impl" @FSM_impl(%a, %b, %c), clock %clk, reset %rst : (i1, i1, i1) -> (i1, i1, i1)
# CHECK:          msft.output %0#0, %0#1, %0#2 : i1, i1, i1
# CHECK:        }


@fsm.machine()
class FSM:
  a = Input(types.i1)
  b = Input(types.i1)
  c = Input(types.i1)

  # States
  A = fsm.State(initial=True)
  (B, C) = fsm.States(2)

  # Transitions
  A.set_transitions((B, lambda ports: ports.a))
  B.set_transitions((A, lambda ports: ports.b), (C,))
  C.set_transitions((B, lambda ports: ports.a))


@unittestmodule()
class FSMUser(Module):
  a = Input(types.i1)
  b = Input(types.i1)
  c = Input(types.i1)
  clk = Input(types.i1)
  rst = Input(types.i1)
  is_a = Output(types.i1)
  is_b = Output(types.i1)
  is_c = Output(types.i1)

  @generator
  def construct(ports):
    fsm = FSM(a=ports.a, b=ports.b, c=ports.c, clk=ports.clk, rst=ports.rst)
    ports.is_a = fsm.is_A
    ports.is_b = fsm.is_B
    ports.is_c = fsm.is_C


# -----

# FSM state transitions example

# CHECK:      fsm.machine @F0_impl(%arg0: i1, %arg1: i1, %arg2: i1) -> (i1, i1, i1, i1) attributes {clock_name = "clock", in_names = ["a", "b", "c"], initialState = "idle", out_names = ["is_A", "is_B", "is_C", "is_idle"], reset_name = "rst"} {
# CHECK-NEXT:    fsm.state @A output {
# CHECK-NEXT:      %true = hw.constant true
# CHECK-NEXT:      %false = hw.constant false
# CHECK-NEXT:      %false_0 = hw.constant false
# CHECK-NEXT:      %false_1 = hw.constant false
# CHECK-NEXT:      fsm.output %true, %false, %false_0, %false_1 : i1, i1, i1, i1
# CHECK-NEXT:    } transitions {
# CHECK-NEXT:      fsm.transition @B guard {
# CHECK-NEXT:        fsm.return %arg0
# CHECK-NEXT:      }
# CHECK-NEXT:    }
# CHECK-NEXT:    fsm.state @B output {
# CHECK-NEXT:      %false = hw.constant false
# CHECK-NEXT:      %true = hw.constant true
# CHECK-NEXT:      %false_0 = hw.constant false
# CHECK-NEXT:      %false_1 = hw.constant false
# CHECK-NEXT:      fsm.output %false, %true, %false_0, %false_1 : i1, i1, i1, i1
# CHECK-NEXT:    } transitions {
# CHECK-NEXT:      fsm.transition @C guard {
# CHECK-NEXT:        %0 = comb.and bin %arg0, %arg1 : i1
# CHECK-NEXT:        %true = hw.constant true
# CHECK-NEXT:        %1 = comb.xor bin %0, %true : i1
# CHECK-NEXT:        %2 = comb.and bin %arg1, %arg2 : i1
# CHECK-NEXT:        %true_0 = hw.constant true
# CHECK-NEXT:        %3 = comb.xor bin %2, %true_0 : i1
# CHECK-NEXT:        %4 = comb.and bin %arg0, %arg2 : i1
# CHECK-NEXT:        %true_1 = hw.constant true
# CHECK-NEXT:        %5 = comb.xor bin %4, %true_1 : i1
# CHECK-NEXT:        %6 = comb.and bin %1, %3, %5 : i1
# CHECK-NEXT:        %true_2 = hw.constant true
# CHECK-NEXT:        %7 = comb.xor bin %6, %true_2 : i1
# CHECK-NEXT:        fsm.return %7
# CHECK-NEXT:      }
# CHECK-NEXT:    }
# CHECK-NEXT:    fsm.state @C output {
# CHECK-NEXT:      %false = hw.constant false
# CHECK-NEXT:      %false_0 = hw.constant false
# CHECK-NEXT:      %true = hw.constant true
# CHECK-NEXT:      %false_1 = hw.constant false
# CHECK-NEXT:      fsm.output %false, %false_0, %true, %false_1 : i1, i1, i1, i1
# CHECK-NEXT:    } transitions {
# CHECK-NEXT:      fsm.transition @idle guard {
# CHECK-NEXT:        fsm.return %arg2
# CHECK-NEXT:      }
# CHECK-NEXT:      fsm.transition @A guard {
# CHECK-NEXT:        %true = hw.constant true
# CHECK-NEXT:        %0 = comb.xor bin %arg1, %true : i1
# CHECK-NEXT:        fsm.return %0
# CHECK-NEXT:      }
# CHECK-NEXT:    }
# CHECK-NEXT:    fsm.state @idle output {
# CHECK-NEXT:      %false = hw.constant false
# CHECK-NEXT:      %false_0 = hw.constant false
# CHECK-NEXT:      %false_1 = hw.constant false
# CHECK-NEXT:      %true = hw.constant true
# CHECK-NEXT:      fsm.output %false, %false_0, %false_1, %true : i1, i1, i1, i1
# CHECK-NEXT:    } transitions {
# CHECK-NEXT:      fsm.transition @A
# CHECK-NEXT:    }
# CHECK-NEXT:  }


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

  idle = fsm.State(initial=True)
  (A, B, C) = fsm.States(3)

  idle.set_transitions((A,))
  A.set_transitions((B, lambda ports: ports.a))
  B.set_transitions((C, maj3))
  C.set_transitions((idle, lambda ports: ports.c),
                    (A, lambda ports: comb.XorOp(ports.b, types.i1(1))))


system = System([F0])
system.generate()
system.print()

# -----

# Shorthand FSM generator.

# CHECK:      fsm.machine @Generated_FSM_impl(%arg0: i1) -> (i1, i1, i1) attributes {clock_name = "clk", in_names = ["go"], initialState = "a", out_names = ["is_a", "is_b", "is_c"], reset_name = "rst"} {
# CHECK-NEXT:    fsm.state @a output {
# CHECK-NEXT:      %true = hw.constant true
# CHECK-NEXT:      %false = hw.constant false
# CHECK-NEXT:      %false_0 = hw.constant false
# CHECK-NEXT:      fsm.output %true, %false, %false_0 : i1, i1, i1
# CHECK-NEXT:    } transitions {
# CHECK-NEXT:      fsm.transition @b
# CHECK-NEXT:      fsm.transition @c guard {
# CHECK-NEXT:        fsm.return %arg0
# CHECK-NEXT:      }
# CHECK-NEXT:    }
# CHECK-NEXT:    fsm.state @b output {
# CHECK-NEXT:      %false = hw.constant false
# CHECK-NEXT:      %true = hw.constant true
# CHECK-NEXT:      %false_0 = hw.constant false
# CHECK-NEXT:      fsm.output %false, %true, %false_0 : i1, i1, i1
# CHECK-NEXT:    } transitions {
# CHECK-NEXT:    }
# CHECK-NEXT:    fsm.state @c output {
# CHECK-NEXT:      %false = hw.constant false
# CHECK-NEXT:      %false_0 = hw.constant false
# CHECK-NEXT:      %true = hw.constant true
# CHECK-NEXT:      fsm.output %false, %false_0, %true : i1, i1, i1
# CHECK-NEXT:    } transitions {
# CHECK-NEXT:    }
# CHECK-NEXT:  }


@unittestmodule()
class FSMUser(Module):
  go = Input(types.i1)
  clk = Input(types.i1)
  rst = Input(types.i1)
  is_a = Output(types.i1)
  is_b = Output(types.i1)
  is_c = Output(types.i1)

  @generator
  def construct(ports):
    MyFSM = fsm.gen_fsm({
        "a": [
            "b",
            ("c", "go"),
        ],
        "b": [],
        "c": []
    }, "Generated_FSM")

    inst = MyFSM(go=ports.go, clk=ports.clk, rst=ports.rst)
    ports.is_a = inst.is_a
    ports.is_b = inst.is_b
    ports.is_c = inst.is_c
