#  Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
#  See https://llvm.org/LICENSE.txt for license information.
#  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from .circt import ir
from .core import CodeGenRoot
from .rtg import rtg
from .labels import Label
from .support import _FromCirctValue


class Test(CodeGenRoot):
  """
  Represents an RTG Test. Stores the test function and location.
  """

  def __init__(self, test_func, args: list[tuple[str, ir.Type]]):
    self.test_func = test_func
    self.arg_names = [name for name, _ in args]
    self.arg_types = [ty for _, ty in args]
    self.success_lbl = None
    self.failure_lbl = dict()

  @property
  def name(self) -> str:
    return self.test_func.__name__

  def _codegen(self):
    # Sort arguments by name
    name_type_pairs = list(enumerate(zip(self.arg_names, self.arg_types)))
    name_type_pairs.sort(key=lambda x: x[1][0])

    test = rtg.TestOp(
        self.name,
        ir.TypeAttr.get(
            rtg.DictType.get([(ir.StringAttr.get(name), ty)
                              for _, (name, ty) in name_type_pairs])))
    block = ir.Block.create_at_start(test.bodyRegion, self.arg_types)
    with ir.InsertionPoint(block):
      # Make sure we apply the inverse permutation to the block arguments to pass them to the python function at the right indices.
      sorted_args = [None] * len(block.arguments)
      for arg, (i, _) in zip(block.arguments, name_type_pairs):
        sorted_args[i] = arg

      self.test_func(*[_FromCirctValue(arg) for arg in sorted_args])
      self.fail("Reached end of test without result status.")
      if self.success_lbl:
        self.success_lbl.place()
        self.succeed()
      for msg, lbl in self.failure_lbl.items():
        lbl.place()
        self.fail(msg)

  def succeed(self) -> None:
    """
    Exit this test and report a success.
    """

    rtg.TestSuccessOp()

  def fail(self, message: str) -> None:
    """
    Exit this test and report a failure with the provided error message.
    """

    rtg.TestFailureOp(message)

  def success_label(self) -> Label:
    """
    A label that can be jumped to to exit the test and report a success.
    """

    if not self.success_lbl:
      self.success_lbl = Label.declare_unique("test_success")

    return self.success_lbl

  def failure_label(self, message: str) -> Label:
    """
    A label that can be jumped to to exit the test and report a failure with
    the provided error message.
    """

    if message not in self.failure_lbl:
      self.failure_lbl[message] = Label.declare_unique("test_failure")

    return self.failure_lbl[message]


def test(*args, **kwargs):
  """
  Decorator for RTG test functions.
  """

  def wrapper(func):
    return Test(func, list(args))

  return wrapper
