#  Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
#  See https://llvm.org/LICENSE.txt for license information.
#  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import dis
import sys
from functools import lru_cache

# Instructions that don't represent meaningful operations and should be skipped
# when scanning forward from a CALL instruction.
_SKIP_OPS = frozenset(("CACHE", "NOP", "PRECALL"))

# Instructions that represent a simple variable store (local, global-as-name,
# or closure variable).
_STORE_OPS = frozenset(("STORE_FAST", "STORE_NAME", "STORE_DEREF"))

# Instructions that load a variable (local, global, closure, or module-level).
_LOAD_OPS = frozenset(
    ("LOAD_FAST", "LOAD_GLOBAL", "LOAD_DEREF", "LOAD_NAME", "LOAD_FAST_CHECK",
     "LOAD_FAST_BORROW"))

# Instructions that load a constant value (int/str literal for subscripts).
_LOAD_CONST_OPS = frozenset(("LOAD_CONST", "LOAD_SMALL_INT"))

# Combined load instructions (Python 3.13+) that load two values at once.
_COMBINED_LOAD_OPS = frozenset(("LOAD_FAST_LOAD_FAST",
                                "LOAD_FAST_BORROW_LOAD_FAST_BORROW"))

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


@lru_cache(maxsize=128)
def _get_instructions(code):
  """Return the list of bytecode instructions for a code object (cached)."""
  return list(dis.get_instructions(code))


def _is_pycde_internal(module_name: str) -> bool:
  """Return True if module_name belongs to pycde but is NOT on the allowlist."""
  if module_name == 'pycde' or module_name.startswith('pycde.'):
    return module_name not in _PYCDE_AUTONAME_MODULES
  return False


def get_var_name(depth=1, skip_pycde=False):
  """Inspect the bytecode of the calling frame (at the given stack depth) to
  determine whether the result of the current call is being assigned to a
  simple local variable. Returns the variable name as a string, or None.

  This deliberately returns None for attribute stores, subscript stores, tuple
  unpacking, and any other non-trivial assignment target.

  If skip_pycde is True, returns None when the target frame belongs to pycde
  internal code (but NOT for modules on the allowlist like pycde.constructs
  and pycde.bsp.*).

  Silently returns None on any failure (non-CPython, unexpected bytecode,
  etc.)."""
  try:
    frame = sys._getframe(depth + 1)

    if skip_pycde:
      module_name = frame.f_globals.get('__name__', '')
      if _is_pycde_internal(module_name):
        return None

    lasti = frame.f_lasti
    instructions = _get_instructions(frame.f_code)

    # Find the instruction at or just before f_lasti.  In Python 3.11-3.12,
    # f_lasti can point into a CACHE instruction gap that dis.get_instructions()
    # hides by default, so we find the last real instruction at offset <= lasti
    # and scan forward from the one after it.
    call_idx = None
    for i, inst in enumerate(instructions):
      if inst.offset <= lasti:
        call_idx = i
      else:
        break
    if call_idx is not None:
      # Scan forward for the first meaningful instruction after the CALL.
      for j in range(call_idx + 1, len(instructions)):
          next_inst = instructions[j]
          if next_inst.opname in _STORE_OPS:
            name = next_inst.argval
            # Exclude the bare "_" throwaway variable.
            if name == "_":
              return None
            return name
          if next_inst.opname in _SKIP_OPS:
            continue
          # Check for subscript store: LOAD (container) LOAD (key) STORE_SUBSCR
          # Also handle combined instructions like LOAD_FAST_LOAD_FAST (3.13+).
          if next_inst.opname in _LOAD_OPS:
            return _try_subscript_name(instructions, j)
          if next_inst.opname in _COMBINED_LOAD_OPS:
            return _try_combined_subscript_name(next_inst, instructions, j)
          # Any other instruction means this isn't a simple assignment.
          return None
    return None
  except Exception:
    return None


def _try_subscript_name(instructions, container_idx):
  """Try to extract a name from a subscript assignment pattern:
  LOAD (container) LOAD/LOAD_CONST (key) STORE_SUBSCR → "container_key".
  Returns the composed name or None."""
  try:
    container_inst = instructions[container_idx]
    container_name = container_inst.argval
    if not isinstance(container_name, str):
      return None

    # Scan past any CACHE/NOP to find the key instruction.
    for k in range(container_idx + 1, len(instructions)):
      key_inst = instructions[k]
      if key_inst.opname in _SKIP_OPS:
        continue
      if key_inst.opname in _LOAD_CONST_OPS:
        # int or string literal subscript
        key_val = key_inst.argval
        if isinstance(key_val, (int, str)):
          return _compose_subscript_name(container_name, key_val, instructions,
                                         k)
        return None
      if key_inst.opname in _LOAD_OPS:
        # Variable subscript — use the variable name
        key_name = key_inst.argval
        if isinstance(key_name, str):
          return _compose_subscript_name(container_name, key_name, instructions,
                                         k)
        return None
      return None
    return None
  except Exception:
    return None


def _compose_subscript_name(container, key, instructions, key_idx):
  """Verify the next meaningful instruction after the key is STORE_SUBSCR,
  then return 'container_key'."""
  for m in range(key_idx + 1, len(instructions)):
    inst = instructions[m]
    if inst.opname in _SKIP_OPS:
      continue
    if inst.opname == "STORE_SUBSCR":
      return f"{container}_{key}"
    return None
  return None


def _try_combined_subscript_name(combined_inst, instructions, idx):
  """Handle combined load instructions like LOAD_FAST_LOAD_FAST (Python 3.13+)
  where argval is a tuple (container_name, key_name), followed by
  STORE_SUBSCR."""
  try:
    names = combined_inst.argval
    if not isinstance(names, tuple) or len(names) != 2:
      return None
    container_name, key_name = names
    if not isinstance(container_name, str) or not isinstance(key_name, str):
      return None
    return _compose_subscript_name(container_name, key_name, instructions, idx)
  except Exception:
    return None
