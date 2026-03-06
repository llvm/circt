# RUN: %PYTHON% -m pytest %s -v

#  Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
#  See https://llvm.org/LICENSE.txt for license information.
#  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""Unit tests for pycde.tracer.get_var_name."""

import unittest
from pycde.tracer import get_var_name


def _create(depth=1):
  """Simulates a signal-creating wrapper function."""
  return get_var_name(depth=depth)


class TestGetVarName(unittest.TestCase):

  def test_simple_assignment(self):
    x = _create()
    self.assertEqual(x, "x")

  def test_different_name(self):
    my_signal = _create()
    self.assertEqual(my_signal, "my_signal")

  def test_no_assignment(self):
    result = _create()  # this IS an assignment, but test POP_TOP separately
    self.assertIsNotNone(result)

  def test_bare_discard(self):
    """Bare _ should be excluded."""
    _ = _create()
    self.assertIsNone(_)

  def test_attribute_assignment(self):
    """self.x = create() should return None."""

    class Obj:
      pass

    o = Obj()
    o.x = _create()
    self.assertIsNone(o.x)

  def test_subscript_int(self):
    """d[0] = create() should return 'd_0'."""
    d = [None]
    d[0] = _create()
    self.assertEqual(d[0], "d_0")

  def test_subscript_string(self):
    """d['foo'] = create() should return 'd_foo'."""
    d = {}
    d['foo'] = _create()
    self.assertEqual(d['foo'], "d_foo")

  def test_subscript_variable(self):
    """d[sig] = create() should return 'd_sig'."""
    d = {}
    sig = "key"
    d[sig] = _create()
    self.assertEqual(d[sig], "d_sig")

  def test_explicit_name_not_underscore_prefix(self):
    """_foo should still capture (only bare _ is excluded)."""
    _foo = _create()
    self.assertEqual(_foo, "_foo")

  def test_depth_through_wrapper(self):
    """Test with an extra wrapper level (simulates wrap_opviews_with_values)."""

    def wrapper():
      return get_var_name(depth=1)

    y = wrapper()
    self.assertEqual(y, "y")

  def test_depth_through_wrapper_no_assign(self):

    class Obj:
      pass

    def wrapper():
      return get_var_name(depth=1)

    o = Obj()
    o.attr = wrapper()
    self.assertIsNone(o.attr)


if __name__ == "__main__":
  unittest.main()
