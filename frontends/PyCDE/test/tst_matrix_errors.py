# RUN: py-split-input-file %s | FileCheck %s

# PSIF: HEADER START
from pycde import System, Input, module, generator
from pycde.pycde_types import types, dim
from pycde.matrix import Matrix
# PSIF: HEADER END

# Missing assignment


@module
class M1:
  in1 = Input(types.i32)

  @generator
  def build(ports):
    m = Matrix((10), dtype=types.i32, name='m1')
    for i in range(9):
      m[i] = ports.in1
    # CHECK: ValueError: Unassigned sub-matrices: (array([9]),)
    m.to_circt()


System([M1]).generate()

# -----

# dtype mismatch


@module
class M1:
  in1 = Input(types.i33)

  @generator
  def build(ports):
    m = Matrix((32), dtype=types.i32, name='m1')
    # CHECK: ValueError: Width mismatch between provided BitVectorValue and target shape.
    m[0] = ports.in1


System([M1]).generate()

# -----

# Invalid constructor


@module
class M1:
  in1 = Input(dim(types.i32, 10))

  @generator
  def build(ports):
    # CHECK: ValueError: Must specify either shape and dtype, or initialize from a value, but not both.
    Matrix((10, 32), from_value=ports.in1, dtype=types.i1, name='m1')


System([M1]).generate()

# -----

# Cast mismatch


@module
class M1:
  in1 = Input(types.i31)

  @generator
  def build(ports):
    m = Matrix((32, 32), dtype=types.i1, name='m1')
    # CHECK: ValueError: Width mismatch between provided BitVectorValue and target shape.
    m[0] = ports.in1


System([M1]).generate()
