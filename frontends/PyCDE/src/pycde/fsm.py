from pycde import System, Input, Output, module, generator
from pycde.module import _SpecializedModule, Generator, _GeneratorPortAccess
from pycde.dialects import fsm
from pycde.pycde_types import types
from typing import Callable

from mlir.ir import InsertionPoint
from collections import defaultdict
from functools import lru_cache

from pycde.support import _obj_to_attribute, attributes_of_type
from circt.support import connect


class State:

  __slots__ = ['initial', 'transitions', 'name', 'output']

  class Transition:

    __slots__ = ['to_state', 'condition']

    def __init__(self, to_state, condition: Callable = None):
      if not isinstance(to_state, State):
        raise ValueError(
            f"to_state must be of State type but got {type(to_state)}")

      self.to_state = to_state
      self.condition = condition

    def _emit(self, state_op, ports):
      with InsertionPoint(state_op.transitions):
        op = fsm.TransitionOp(self.to_state.name)

        # If a condition function was specified, execute it on the ports and
        # assign the result as the guard of this transition.
        if self.condition:
          op.set_guard(lambda: self.condition(ports))

  def __init__(self, initial=False):
    self.initial = initial
    self.transitions = []
    self.name = None

    # A handle to the output port indicating that this state is active.
    self.output = None

  def set_transitions(self, *transitions):
    self.transitions = [State.Transition(*t) for t in transitions]

  def _emit(self, spec_mod, ports):
    # Create state op
    assert self.name is not None
    state_op = fsm.StateOp(self.name)

    # Assign the current state as being active in the FSM output vector.
    with InsertionPoint(state_op.output):
      outputs = []
      for (outport_it_name, _) in spec_mod.output_ports:
        outputs.append(types.i1(outport_it_name == self.output.name))
      fsm.OutputOp(*outputs)

    # Emit outgoing transitions from this state.
    for transition in self.transitions:
      transition._emit(state_op, ports)


def States(n):
  """ Utility function to generate multiple states. """
  return [State() for _ in range(n)]


def create_fsm_machine_op(sys, mod: _SpecializedModule, symbol):
  """Creation callback for creating a FSM MachineOp."""

  # Add attributes for in- and output names.
  attributes = {}
  attributes["in_names"] = _obj_to_attribute(
      [port_name for port_name, _ in mod.input_ports])
  attributes["out_names"] = _obj_to_attribute(
      [port_name for port_name, _ in mod.output_ports])

  # Add attributes for clock and reset names.
  attributes["clock_name"] = _obj_to_attribute(mod.modcls.clock_name)
  if mod.modcls.reset_name:
    attributes["reset_name"] = _obj_to_attribute(mod.modcls.reset_name)

  return fsm.MachineOp(symbol,
                       mod.modcls._initial_state,
                       mod.input_ports,
                       mod.output_ports,
                       attributes=attributes,
                       loc=mod.loc,
                       ip=sys._get_ip())


def generate_fsm_machine_op(generate_obj: Generator,
                            spec_mod: _SpecializedModule):
  """ Generator callback for generating an FSM op. """
  entry_block = spec_mod.circt_mod.body.blocks[0]
  ports = _GeneratorPortAccess(spec_mod)

  with InsertionPoint(entry_block), generate_obj.loc:
    for state in spec_mod.modcls.states:
      state._emit(spec_mod, ports)


def machine(clock: str = 'clk', reset: str = None):
  """
  Top-level FSM decorator which gives the user the option specify port
  names for the clock and (optional) reset signal.
  These port names will be used in the FSM HW module wrapper, as well as
  carried through as attributes on the fsm.machine operation to inform
  FSM HW lowering about desired port names.
  """

  def machine_clocked(to_be_wrapped):
    """
    Wrap a class as an FSM machine.

    An FSM PyCDE module is expected to implement:

    - A set of input ports:
        These can be of any type, and are used to drive the FSM.

        Transitions can be specified either as a tuple of (next_state, condition)
        or as a single `next_state` string (unconditionally taken).
        a `condition` function is a function which takes a single input representing
        the `ports` of a component, similar to the `@generator` decorator used
        elsewhere in PyCDE.
    """
    states = {}
    initial_state = None
    for name, v in attributes_of_type(to_be_wrapped, State).items():
      if name in states:
        raise ValueError("Duplicate state name: {}".format(name))
      v.name = name
      states[name] = v
      if v.initial:
        if initial_state is not None:
          raise ValueError(
              f"Multiple initial states specified ({name}, {initial_state})")
        initial_state = name

    if initial_state is None:
      raise ValueError(
          "No initial state specified, please create a state with `initial=True`"
      )

    # At this point, the 'states' attribute should be considered an immutable,
    # ordered list of states.
    to_be_wrapped.states = states.values()
    to_be_wrapped._initial_state = initial_state

    if len(states) == 0:
      raise ValueError("No States defined")

    # Add an output port for each state.
    for state_name, state in states.items():
      output_for_state = Output(types.i1)
      setattr(to_be_wrapped, 'is_' + state_name, output_for_state)
      state.output = output_for_state

    # Store requested clock and reset names.
    to_be_wrapped.clock_name = clock
    to_be_wrapped.reset_name = reset

    # Set module creation and generation callbacks.
    setattr(to_be_wrapped, 'create_cb', create_fsm_machine_op)
    setattr(to_be_wrapped, 'generator_cb', generate_fsm_machine_op)

    # Create a dummy Generator function to trigger module generation.
    # This function doesn't do anything, since all generation logic is embedded
    # within generate_fsm_machine_op. In the future, we may allow an actual
    # @generator function specified by the user if they want to do something
    # specific.
    setattr(to_be_wrapped, 'dummy_generator_f', generator(lambda x: None))

    # Treat the remainder of the class as a module.
    # Rename the fsm_mod before creating the module to ensure that the wrapped
    # module will be named as the user specified (that of fsm_mod), and the
    # FSM itself will have a suffixed name.
    fsm_name = to_be_wrapped.__name__
    to_be_wrapped.__name__ = to_be_wrapped.__name__ + '_impl'
    to_be_wrapped.__qualname__ = to_be_wrapped.__qualname__ + '_impl'
    fsm_mod = module(to_be_wrapped)

    # Next we build the outer wrapper class that contains clock and reset ports.
    fsm_hw_mod = fsm_wrapper_class(fsm_mod=fsm_mod,
                                   fsm_name=fsm_name,
                                   clock=clock,
                                   reset=reset)

    return module(fsm_hw_mod)

  return machine_clocked


def fsm_wrapper_class(fsm_mod, fsm_name, clock, reset=None):
  """
  Generate a wrapper class for the FSM class which contains the clock and reset
  signals, as well as a `fsm.hw_instance` instaitiation of the FSM.
  """

  class fsm_hw_mod:

    @generator
    def construct(ports):
      in_ports = {
          port_name: getattr(ports, port_name)
          for (port_name, _) in fsm_mod._pycde_mod.input_ports
      }
      fsm_instance = fsm_mod(**in_ports)
      # Attach clock and optional reset on the backedges created during
      # the MachineOp:instatiate call.
      clock_be = getattr(fsm_instance._instantiation, '_clock_backedge')
      connect(clock_be, getattr(ports, clock))
      if hasattr(fsm_instance._instantiation, '_reset_backedge'):
        reset_be = getattr(fsm_instance._instantiation, '_reset_backedge')
        connect(reset_be, getattr(ports, reset))

      # Connect outputs
      for (port_name, _) in fsm_mod._pycde_mod.output_ports:
        setattr(ports, port_name, getattr(fsm_instance, port_name))

  # Inherit in and output ports. We do this outside of the wrapped class
  # since we cannot do setattr inside the class scope (def-use).
  for (name, type) in fsm_mod._pycde_mod.input_ports:
    setattr(fsm_hw_mod, name, Input(type))
  for (name, type) in fsm_mod._pycde_mod.output_ports:
    setattr(fsm_hw_mod, name, Output(type))

  # Add clock and additional reset port.
  setattr(fsm_hw_mod, clock, Input(types.i1))

  if reset is not None:
    setattr(fsm_hw_mod, reset, Input(types.i1))

  # The wrapper class now overloads the name of the user-defined FSM.
  # From this point on, instantiating the user FSM class will actually
  # instantiate the wrapper HW module class.
  fsm_hw_mod.__qualname__ = fsm_name
  fsm_hw_mod.__name__ = fsm_name
  fsm_hw_mod.__module__ = fsm_name
  return fsm_hw_mod
