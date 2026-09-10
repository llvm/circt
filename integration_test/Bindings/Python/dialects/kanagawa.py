# REQUIRES: bindings_python
# RUN: %PYTHON% %s

import circt

from circt.ir import Context, Module
from circt import passmanager

with Context() as ctx:
  circt.register_dialects(ctx)
  mod = Module.parse("""
kanagawa.design @foo {
  kanagawa.container sym @C {
  }

  kanagawa.container sym @AccessChild {
    %c = kanagawa.container.instance @c, <@foo::@C>
    %c_ref = kanagawa.path [
      #kanagawa.step<child , @c : !kanagawa.scoperef<@foo::@C>>
    ]
  }
}
""")

  # Test that we can parse kanagawa dialect IR
  print("Kanagawa dialect registration and parsing successful!")

  # Test that kanagawa passes are registered
  pm = passmanager.PassManager.parse(
      "builtin.module(kanagawa.design(kanagawa-containerize))")
  print("Kanagawa passes registration successful!")
