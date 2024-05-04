from hmac import new
from .common import Input, Output
from .dialects import fsm
from .module import Module, ModuleLikeBuilderBase
from .support import _obj_to_attribute
from .types import Bits, types

from .circt.ir import FlatSymbolRefAttr, InsertionPoint, StringAttr
from .circt.support import attribute_to_var
from .circt.dialects import fsm as raw_fsm

from typing import Callable, Set


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

    def _emit(self, ports):
      op = fsm.TransitionOp(self.to_state.name)

      # If a condition function was specified, execute it on the ports and
      # assign the result as the guard of this transition.
      if self.condition:
        op.set_guard(lambda: self.condition(ports).value)

  def __init__(self, initial=False):
    self.initial = initial
    self.transitions = []
    self.name = None

    # A handle to the output port indicating that this state is active.
    self.output = None

  def set_transitions(self, *transitions):
    self.transitions = [State.Transition(*t) for t in transitions]

  def add_transitions(self, *transitions):
    self.transitions.extend([State.Transition(*t) for t in transitions])

  def _emit(self, spec_mod, ports):
    # Create state op
    assert self.name is not None
    state_op = fsm.StateOp(self.name)

    # Assign the current state as being active in the FSM output vector.
    with InsertionPoint(state_op.output):
      outputs = []
      for idx in range(len(spec_mod.outputs)):
        outputs.append(types.i1(idx == self.output))
      fsm.OutputOp(*outputs)

    # Emit outgoing transitions from this state.
    with InsertionPoint(state_op.transitions):
      for transition in self.transitions:
        transition._emit(ports)


def States(n):
  """ Utility function to generate multiple states. """
  return [State() for _ in range(n)]


class MachineModuleBuilder(ModuleLikeBuilderBase):
  """Define how to build an FSM."""

  @property
  def circt_mod(self):
    """Get the raw CIRCT operation for the module definition. DO NOT store the
    returned value!!! It needs to get reaped after the current action (e.g.
    instantiation, generation). Memory safety when interacting with native code
    can be painful."""

    from .system import System
    sys: System = System.current()
    ret = sys._op_cache.get_circt_mod(self)
    if ret is None:
      return sys._create_circt_mod(self)
    return ret

  def scan_cls(self):
    """Scan the class as usual, but also scan for State. Add implicit signals.
    Run some other validity checks."""
    super().scan_cls()

    states = {}
    initial_state = None
    for name, v in self.cls_dct.items():
      if not isinstance(v, State):
        continue

      if name in states:
        raise ValueError("Duplicate state name: {}".format(name))
      v.name = name
      states[name] = v
      if v.initial:
        if initial_state is not None:
          raise ValueError(
              f"Multiple initial states specified ({name}, {initial_state}).")
        initial_state = name

    from .types import ClockType
    for port in self.inputs:
      if not (isinstance(port.type, ClockType) or
              (hasattr(port.type, "width") and port.type.width == 1)):
        raise ValueError(
            f"Input port {port.name} has width {port.type.width}. For now, FSMs only "
            "support i1 inputs.")

    # At this point, the 'states' attribute should be considered an immutable,
    # ordered list of states.
    self.states = states.values()
    self.initial_state = initial_state

    if len(states) > 0 and self.initial_state is None:
      raise ValueError("No initial state specified, please create a state with "
                       "`initial=True`.")

    # Add an output port for each state.
    num_outputs = len(self.outputs)
    for state_name, state in states.items():
      state.output = len(self.outputs)
      o = Output(Bits(1), name="is_" + state_name)
      o.idx = num_outputs
      num_outputs += 1
      setattr(self.modcls, o.name, o)
      self.ports.append(o)

    inputs_to_remove: Set[Input] = set()
    if len(self.clocks) > 1:
      raise ValueError("FSMs must have at most one clock")
    else:
      self.clock_name = "clk"
      if len(self.clocks) == 1:
        idx = self.clocks.pop()
        clock_port = self.inputs[idx]
        self.clock_name = clock_port.name
        inputs_to_remove.add(clock_port)

    if len(self.resets) > 1:
      raise ValueError("FSMs must have at most one reset")
    else:
      self.reset_name = "rst"
      if len(self.resets) == 1:
        idx = self.resets.pop()
        reset_port = self.inputs[idx]
        self.reset_name = reset_port.name
        inputs_to_remove.add(reset_port)

    # Remove the clock and reset inputs, if necessary.
    new_ports = []
    new_num_inputs = 0
    for port in self.ports:
      if not isinstance(port, Input):
        new_ports.append(port)
      else:
        if port in inputs_to_remove:
          port.idx = None
        else:
          port.idx = new_num_inputs
          new_num_inputs += 1
          new_ports.append(port)
    self.ports = new_ports

  def create_op(self, sys, symbol):
    """Creation callback for creating a FSM MachineOp."""

    if len(self.states) == 0:
      raise ValueError("No States defined")

    # Add attributes for in- and output names.
    attributes = {}
    attributes["in_names"] = _obj_to_attribute(
        [port.name for port in self.inputs])
    attributes["out_names"] = _obj_to_attribute(
        [port.name for port in self.outputs])

    # Add attributes for clock and reset names.
    attributes["clock_name"] = _obj_to_attribute(self.clock_name)
    attributes["reset_name"] = _obj_to_attribute(self.reset_name)

    machine_op = fsm.MachineOp(symbol,
                               self.initial_state,
                               [(p.name, p.type._type) for p in self.inputs],
                               [(p.name, p.type._type) for p in self.outputs],
                               attributes=attributes,
                               loc=self.loc,
                               ip=sys._get_ip())

    entry_block = machine_op.body.blocks[0]
    ports = self.generator_port_proxy(entry_block.arguments, self)

    with self.GeneratorCtxt(self, ports, entry_block, self.loc):
      for state in self.states:
        state._emit(self, ports)

    return machine_op

  def instantiate(self, impl, kwargs, instance_name: str):
    circt_mod = self.circt_mod

    in_names = attribute_to_var(circt_mod.attributes['in_names'])
    inputs = [kwargs[port].value for port in in_names]

    # Clock and resets are not part of the input ports of the FSM, but
    # it is at the point of `fsm.hw_instance` instantiation that they
    # must be connected.
    clock = kwargs[StringAttr(circt_mod.attributes['clock_name']).value]
    reset = kwargs[StringAttr(circt_mod.attributes['reset_name']).value]

    op = raw_fsm.HWInstanceOp(outputs=circt_mod.type.results,
                              inputs=inputs,
                              name=StringAttr.get(instance_name),
                              machine=FlatSymbolRefAttr.get(
                                  StringAttr(
                                      circt_mod.attributes["sym_name"]).value),
                              clock=clock.value,
                              reset=reset.value)
    return op


class Machine(Module):
  """Base class to be extended for defining an FSM."""

  BuilderType = MachineModuleBuilder
