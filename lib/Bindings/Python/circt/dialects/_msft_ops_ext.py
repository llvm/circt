#  Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
#  See https://llvm.org/LICENSE.txt for license information.
#  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from circt.dialects import msft as _msft
import circt.dialects._hw_ops_ext as _hw_ext
import circt.support as support

import mlir.ir as _ir


class InstanceBuilder(support.NamedValueOpView):
  """Helper class to incrementally construct an instance of a module."""

  def __init__(self,
               module,
               name,
               input_port_mapping,
               *,
               sym_name=None,
               loc=None,
               ip=None):
    self.module = module
    instance_name = _ir.StringAttr.get(name)
    module_name = _ir.FlatSymbolRefAttr.get(_ir.StringAttr(module.name).value)
    if sym_name:
      sym_name = _ir.StringAttr.get(sym_name)
    pre_args = [instance_name, module_name]
    results = module.type.results

    super().__init__(_msft.InstanceOp,
                     results,
                     input_port_mapping,
                     pre_args, [],
                     loc=loc,
                     ip=ip)

  def create_default_value(self, index, data_type, arg_name):
    type = self.module.type.inputs[index]
    return support.BackedgeBuilder.create(type,
                                          arg_name,
                                          self,
                                          instance_of=self.module)

  def operand_names(self):
    arg_names = _ir.ArrayAttr(self.module.attributes["argNames"])
    arg_name_attrs = map(_ir.StringAttr, arg_names)
    return list(map(lambda s: s.value, arg_name_attrs))

  def result_names(self):
    arg_names = _ir.ArrayAttr(self.module.attributes["resultNames"])
    arg_name_attrs = map(_ir.StringAttr, arg_names)
    return list(map(lambda s: s.value, arg_name_attrs))


class MSFTModuleOp(_hw_ext.ModuleLike):

  def __init__(
      self,
      name,
      input_ports=[],
      output_ports=[],
      parameters: _ir.DictAttr = None,
      loc=None,
      ip=None,
  ):
    attrs = {"parameters": parameters}
    super().__init__(name,
                     input_ports,
                     output_ports,
                     attributes=attrs,
                     loc=loc,
                     ip=ip)

  def create(self, name: str, loc=None, ip=None, **kwargs):
    return InstanceBuilder(self, name, kwargs, loc=loc, ip=ip)

  def add_entry_block(self):
    self.body.blocks.append(*self.type.inputs)
    return self.body.blocks[0]

  @property
  def body(self):
    return self.regions[0]

  @property
  def entry_block(self):
    return self.regions[0].blocks[0]
