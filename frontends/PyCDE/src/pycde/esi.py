from pycde import Input, Output, generator
from pycde.dialects import esi
from pycde.pycde_types import types
from typing import Callable

from mlir.ir import InsertionPoint

from pycde.support import _obj_to_attribute, attributes_of_type
from circt.support import connect

import typing


class ServiceDecl:
  pass


class HostComms(ServiceDecl):
  ToHost = Input(types.any)
  FromHost = Output(types.any)


def Server(cls: typing.Type):
  return cls


@Server
class Cosim:
  pass
