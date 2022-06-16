#  Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
#  See https://llvm.org/LICENSE.txt for license information.
#  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from .value import (BitVectorValue, ListValue)
from .pycde_types import BitVectorType, dim
from pycde.dialects import hw, sv
import numpy as np
import mlir.ir as ir
from functools import lru_cache
from dataclasses import dataclass


@dataclass
class TargetShape:
  # The TargetShape class is a small helper class for representing shapes
  # which might be n-dimensional matrices (len(dims) > 0) or unary types
  # (len(dims) == 0).
  dims: list
  dtype: type

  @property
  def num_dims(self):
    return len(self.dims)

  @property
  def type(self):
    if len(self.dims) != 0:
      return dim(self.dtype, *self.dims)
    else:
      return self.dtype

  def __str__(self):
    return str(self.type)


class Matrix(np.ndarray):
  """
  A PyCDE Matrix serves as a Numpy view of a multidimensional CIRCT array (ArrayType).
  The Matrix ensures that all assignments to itself have been properly converted
  to conform with insertion into the numpy array (circt_to_arr).
  Once filled, a user can treat the Matrix as a numpy array.
  The underlying CIRCT array is not materialized until to_circt is called.
  """

  def __array_finalize__(self, obj):
    # Ensure Matrix-class attributes are propagated upon copying.
    # see https://numpy.org/doc/stable/reference/arrays.classes.html#numpy.class.__array_finalize__
    if obj is not None:
      self.__dict__ = obj.__dict__

  def __new__(cls,
              shape: list = None,
              dtype=None,
              name: str = None,
              from_value=None) -> None:
    """Construct a matrix with the given shape and dtype.
    This is a __new__ function since np.ndarray does not have an __init__ function.
      Args:
        shape: A tuple of integers representing the shape of the matrix.
        dtype: the inner type of the matrix. This is a PyCDE type - the Numpy
               matrix contains 'object'-typed values.
      """

    if bool(from_value) and bool(shape or dtype):
      raise ValueError(
          "Must specify either shape and dtype, or initialize from a value, but not both."
      )

    if from_value is not None:
      if not isinstance(from_value, ListValue):
        raise ValueError("from_value must be a ListValue")
      shape = from_value.type.shape
      dtype = from_value.type.inner_type
      name = from_value.name

    # Initialize the underlying np.ndarray
    self = np.ndarray.__new__(cls, shape=shape, dtype=object)

    if name is None:
      name = "matrix"  # todo: require name?
    self.name = name
    self.pycde_dtype = dtype

    # The SSA value of this matrix after it has been materialized.
    # Once set, the matrix is immutable.
    self.circt_output = None

    if from_value is not None:
      # PyCDE and numpy do not play nicely when doing np.arr(self.circt_to_arr(...))
      # but individual assignments work.
      target_shape = self.target_shape_for_idxs(0)
      value_arr = self.circt_to_arr(from_value, target_shape)
      for i, v in enumerate(value_arr):
        super().__setitem__(self, i, v)

    return self

  @lru_cache
  def get_constant(self, value: int, width: int = 32):
    """ Get an IR constant backed by a constant cache."""
    return hw.ConstantOp(ir.IntegerType.get_signless(width), value)

  def circt_to_arr(self, value, target_shape):
    """Converts a CIRCT value into a numpy array."""
    if isinstance(value, BitVectorValue) and isinstance(target_shape.dtype,
                                                        BitVectorType):
      # Direct match on the target shape, which is not an array?
      if value.type == target_shape.dtype and target_shape.num_dims == 0:
        return value

      # Is it feasible to extract values to the target shape?
      if target_shape.num_dims > 1:
        raise ValueError(
            f"Cannot extract BitVectorValue of type {value.type} to a multi-dimensional array of type {target_shape.type}."
        )

      target_shape_bits = (target_shape.dims[0] if len(target_shape.dims) else
                           1) * target_shape.dtype.width
      if target_shape_bits != value.type.width:
        raise ValueError(
            f"Width mismatch between provided BitVectorValue ({value.type}) and target shape ({target_shape.type})."
        )

      # Extract to the target type
      n = len(value) / target_shape.dtype.width
      if n != int(n) or n < 1:
        raise ValueError("Bitvector must be a multiple of the provided dtype")
      n = int(n)
      slice_elem_width = int(len(value) / n)
      arr = []
      if n == 1:
        return value
      else:
        for i in range(n):
          startbit = i * slice_elem_width
          endbit = i * slice_elem_width + slice_elem_width
          arr.append(value[startbit:endbit])
    elif isinstance(value, ListValue):
      # Recursively convert the list.
      arr = []
      for i in range(value.type.size):
        # Pop the outer dimension of the target shape.
        inner_dims = target_shape.dims.copy()[1:]
        arr.append(
            self.circt_to_arr(value[i],
                              TargetShape(inner_dims, target_shape.dtype)))
    else:
      raise ValueError(f"Cannot convert value {value} to numpy array.")

    return arr

  def target_shape_for_idxs(self, idxs):
    """Get the TargetShape for the given indexing into the array.
    
    This function can be used for determining the type that right-hand side values
    to a given matrix assignment should have.
    """
    target_v = self[idxs]
    target_shape = TargetShape([], self.pycde_dtype)
    if isinstance(target_v, np.ndarray):
      target_shape.dims = list(target_v.shape)
    return target_shape

  def __setitem__(self, np_access, value):
    if self.circt_output is not None:
      raise ValueError("Cannot assign to a materialized matrix.")

    # Todo: We should allow for 1 extra dimension in the access slice which is
    # not passed to numpy. This dimension would instead refer to an access
    # into the inner data type.
    # This access would allow for
    # - bitwise access for integer data types
    # - struct access for structs
    # This issue, however, is a more general one, since we don't currently
    # support lhs assignment of PyCDE Value's, which would be require for this.

    # Infer the target shape based on the access to the numpy array.
    # circt_to_arr will then try to convert the value to this shape.
    v = self.circt_to_arr(value, self.target_shape_for_idxs(np_access))
    super().__setitem__(np_access, v)

  def check_is_fully_assigned(self):
    """ Checks that all sub-matrices have been fully assigned. """
    unassigned = np.where(self == None)
    if len(unassigned[0]) > 0:
      raise ValueError(f"Unassigned sub-matrices: {unassigned}")

  def to_circt(self, create_wire=True, dtype=None):
    """Materializes this matrix to a ListValue through hw.array_create operations.
    
    if 'create_wire' is True, the matrix will be materialized to an sv.wire operation
    and the returned value will be a read-only reference to the wire.
    This wire acts as a barrier in CIRCT to prevent dataflow optimizations
    from reordering/optimizing the materialization of the matrix, which might
    reduce debugability.
    """
    if self.circt_output:
      return self.circt_output

    if dtype == None:
      dtype = self.pycde_dtype

    # Check that the entire matrix has been assigned. If not, an exception is
    # thrown.
    self.check_is_fully_assigned()

    def build_subarray(lstOrVal):
      # Recursively converts this matrix into ListValues through hw.array_create
      # operations.
      if not isinstance(lstOrVal, BitVectorValue):
        subarrays = [build_subarray(v) for v in lstOrVal]
        return hw.ArrayCreateOp(subarrays)
      return lstOrVal

    # Materialize the matrix
    self.circt_output = build_subarray(self)

    if create_wire:
      wire = sv.WireOp(self.circt_output.type, self.name + "_wire")
      sv.AssignOp(wire, self.circt_output)
      self.circt_output = wire.read

    return self.circt_output


"""
Inject a curated list of numpy functions into the ListValue class.
This allows for directly manipulating the ListValues with numpy functionality.
Power-users who use the Matrix directly have access to all numpy functions.
In reality, it will only be a subset of the numpy array functions which are
safe to be used in the PyCDE context. Curating access at the level of ListValues
seems like a safe starting point.
"""
numpy_functions = [
    "transpose", "reshape", "flatten", "moveaxis", "rollaxis", "swapaxes"
]


def apply_numpy_f_to_listvalue(f):

  def wrapper(list_value, *args, **kwargs):
    return getattr(Matrix(from_value=list_value), f)(*args, **kwargs).to_circt()

  return wrapper


for f in numpy_functions:
  setattr(ListValue, f, apply_numpy_f_to_listvalue(f))
