#  Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
#  See https://llvm.org/LICENSE.txt for license information.
#  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import ast
import sys

import executing

# PyCDE-internal modules where auto-naming from variable names IS desired.
# Code in any other pycde.* module is considered "internal plumbing" and will
# be skipped when skip_pycde is True.
_PYCDE_AUTONAME_MODULES = frozenset((
    "pycde.constructs",
    "pycde.bsp",
    "pycde.bsp.common",
    "pycde.bsp.cosim",
    "pycde.bsp.dma",
    "pycde.bsp.xrt",
))


def _is_pycde_internal(module_name: str) -> bool:
  """Return True if module_name belongs to pycde but is NOT on the allowlist."""
  if module_name == 'pycde' or module_name.startswith('pycde.'):
    return module_name not in _PYCDE_AUTONAME_MODULES
  return False


def _name_from_target(target: ast.AST):
  """Extract a name string from an AST assignment target node.

  Returns a name for simple variable assignments and subscript assignments
  with constant or simple-name keys.  Returns None for attribute stores,
  starred targets, tuple unpacking, or anything else we can't map to a
  simple name."""
  if isinstance(target, ast.Name):
    return target.id if target.id != "_" else None

  if isinstance(target, ast.Subscript) and isinstance(target.value, ast.Name):
    container = target.value.id
    s = target.slice
    if isinstance(s, ast.Constant) and isinstance(s.value, (int, str)):
      return f"{container}_{s.value}"
    if isinstance(s, ast.Name):
      return f"{container}_{s.id}"

  return None


def get_var_name(depth=1, skip_pycde=False):
  """Determine whether the result of the current call is being assigned to a
  simple variable.  Returns the variable name as a string, or None.

  Uses the ``executing`` library to identify the currently executing AST
  node and inspects the enclosing statement for an assignment target.

  If skip_pycde is True, returns None when the target frame belongs to pycde
  internal code (but NOT for modules on the allowlist like pycde.constructs
  and pycde.bsp.*).

  Silently returns None on any failure."""
  try:
    frame = sys._getframe(depth + 1)

    if skip_pycde:
      module_name = frame.f_globals.get('__name__', '')
      if _is_pycde_internal(module_name):
        return None

    ex = executing.Source.executing(frame)
    node = ex.node
    if node is None:
      return None
    for stmt in ex.statements:
      if isinstance(stmt, ast.Assign) and len(stmt.targets) == 1:
        # Only return a name if the executing node is the direct RHS of the
        # assignment — not a sub-expression within a larger expression.
        if stmt.value is node:
          return _name_from_target(stmt.targets[0])
    return None
  except Exception:
    return None
