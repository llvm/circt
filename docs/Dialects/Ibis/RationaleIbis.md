# `ibis` Dialect Rationale

## Lowering flow

1. Containerization:
  At this level, one may have relative references to ports `get_port` accesses
  to `!ibis.scoperef`s
2. Tunneling:
  Relative `!ibis.scoperef` references are defined via `ibis.path` operations.
  In this pass, we lower `ibis.path` operations by tunneling `portref`s through
  the instance hierarchy, based on the `get_port` operations that were present
  in the various containers. After this, various `portref<in portref<#dir, T>>`
  ports are present in the design, which represents the actual ports that are
  being passed around.
3. `portref` lowering
  Next, we need to convert `portref<in portref<#dir, T>>`-typed ports into
  `T` typed in- or output ports. We do this by analyzing how a portref is used
  inside a container, and then creating an in- or output port based on that.
  That is:
  - write to `portref<in portref<in, T>>` becomes `out T`
  - read from `portref<in portref<out, T>>` becomes `in T`
  - write to `portref<out portref<out, T>>` becomes `out T` (a port reference
    inside the module will be driven by a value from the outside)
  - read from `portref<out portref<in, T>>` becomes `in T` (a port reference
    inside the module will be driven by a value from the outside)
4. Removal of self-driving inputs:
  In cases where children drive parent ports, the prior case may create
  situations where a container drives its own input ports (and by extension, no
  other instantiating container is expected to drive that port of the instance,
  if the IR is correct). We thus run `ibis-clean-selfdrivers` to replace these
  self-driven ports by the actual values that are being driven into them.
5. HW lowering:
  At this point, there no longer exist any relative references in the Ibis IR,
  and all instantiations should (if the IR is correct) have all of their inputs
  driven. This means that we can now lower the IR to `hw.module`s.
