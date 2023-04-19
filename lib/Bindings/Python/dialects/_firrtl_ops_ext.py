from typing import Sequence

from circt.ir import (ArrayAttr, Attribute, IntegerAttr, IntegerType,
                      InsertionPoint, Location, OpView, StringAttr, Type,
                      TypeAttr, Value)


class HasBody:

  def __init__(self, *args, types: Sequence[Type] = None, **kwargs) -> None:
    super().__init__(*args, **kwargs)
    if types:
      self.body.blocks.append(*types)
    else:
      self.body.blocks.append()

  def __enter__(self) -> "HasBody":
    self.ip = InsertionPoint(self.body.blocks[0])
    self.ip.__enter__()
    return self

  def __exit__(self, *args, **kwargs) -> None:
    self.ip.__exit__(*args, **kwargs)


class CircuitOp(HasBody):

  def __init__(self, name: str) -> None:
    results = []
    operands = []
    attributes = {"name": StringAttr.get(name)}
    super().__init__(
        self.build_generic(results=results,
                           operands=operands,
                           attributes=attributes))


class FModuleOp(HasBody):

  def __init__(self, name: str, **kwargs) -> None:
    from circt.dialects import firrtl

    ports = []
    for (port_name, direction_type) in kwargs.items():
      if not isinstance(direction_type, firrtl.DirectionType):
        continue
      ports.append(
          firrtl.Port(name=port_name,
                      direction=direction_type.direction,
                      type=direction_type.type))

    port_names = [StringAttr.get(p.name) for p in ports]
    port_types = [TypeAttr.get(p.type) for p in ports]
    port_directions = int(
        ''.join(reversed([str(p.direction.value) for p in ports])), 2)
    port_annotations = []
    port_symbols = []
    port_locations = [Location.current.attr for p in ports]

    convention = kwargs.get("convention", "internal")

    results = []
    operands = []
    attributes = {
        "convention":
            firrtl.Convention(convention),
        "sym_name":
            StringAttr.get(name),
        "portNames":
            ArrayAttr.get(port_names),
        "portTypes":
            ArrayAttr.get(port_types),
        "portDirections":
            IntegerAttr.get(IntegerType.get_signless(len(ports)),
                            port_directions),
        "portAnnotations":
            ArrayAttr.get(port_annotations),
        "portSyms":
            ArrayAttr.get(port_symbols),
        "portLocations":
            ArrayAttr.get(port_locations)
    }

    block_arg_types = [p.value for p in port_types]

    super().__init__(self.build_generic(results=results,
                                        operands=operands,
                                        attributes=attributes),
                     types=block_arg_types)

    self._ports = {
        p.name: self.body.blocks[0].arguments[i] for (i, p) in enumerate(ports)
    }

  def __getattr__(self, name: str) -> Value:
    port = self._ports.get(name)
    if port is not None:
      return port

    raise AttributeError(
        f"module {self.attributes['sym_name']} has no port named {name}")


class RegOp:

  def __init__(self, name: str, clock: Value, type: Type) -> None:
    from circt.dialects import firrtl

    super().__init__(
        self.build_generic(results=[type],
                           operands=[clock],
                           attributes={
                               "name": StringAttr.get(name),
                               "nameKind": firrtl.NameKind("interesting_name"),
                               "annotations": ArrayAttr.get([])
                           }))


class ConstantOp:

  def __init__(self, value: int, type: Type) -> None:
    width = int(str(type).split("<")[-1][:-1])
    is_signed = "s" in str(type)
    integer_type = IntegerType.get_signed(
        width) if is_signed else IntegerType.get_unsigned(width)
    OpView.__init__(
        self,
        self.build_generic(
            results=[type],
            operands=[],
            attributes={"value": IntegerAttr.get(integer_type, value)}))
