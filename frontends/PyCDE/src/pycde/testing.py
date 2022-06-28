from pycde import System, module

import builtins
import inspect


def unittestmodule(generate=True,
                   print=True,
                   run_passes=False,
                   emit_outputs=False,
                   **kwargs):
  """
  Like @module, but additionally performs system instantiation, generation,
  and printing to reduce boilerplate in tests.
  In case of wrapping a function, @testmodule accepts kwargs which are passed
  to the function as arguments.
  """

  def testmodule_inner(func_or_class):
    mod = module(func_or_class)

    # Apply any provided kwargs if this was a function.
    if inspect.isfunction(func_or_class):
      mod = mod(**kwargs)

    # Add the module to global scope in case it's referenced within the
    # module generator functions
    setattr(builtins, mod.__name__, mod)

    sys = System([mod])
    if generate:
      sys.generate()
      if print:
        sys.print()
      if run_passes:
        sys.run_passes()
      if emit_outputs:
        sys.emit_outputs()

    return mod

  return testmodule_inner
