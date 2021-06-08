#  Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
#  See https://llvm.org/LICENSE.txt for license information.
#  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception


# There is currently no support in MLIR for querying attribute types. The
# conversation regarding how to achieve this is ongoing and I expect it to be a
# long one. This is a way that works for now.
def attribute_to_var(attr):
  import mlir.ir as ir
  try:
    return ir.BoolAttr(attr).value
  except ValueError:
    pass
  try:
    return ir.IntegerAttr(attr).value
  except ValueError:
    pass
  try:
    return ir.StringAttr(attr).value
  except ValueError:
    pass
  try:
    arr = ir.ArrayAttr(attr)
    return [attribute_to_var(x) for x in arr]
  except ValueError:
    pass

  raise TypeError(f"Cannot convert {repr(attr)} to python value")
