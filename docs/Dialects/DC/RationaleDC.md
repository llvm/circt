# DC Dialect Rationale

[TOC]

## Introduction

DC (**D**ynamic **C**ontrol) IR describes independent, unsynchronized processes communicating data through First-in First-out (FIFO) communication channels. 
This can be implemented in many ways, such as using synchronous logic, or with processors. 

The intention of DC is to model all operations required to represent such a control flow language. DC aims to be strictly a control
flow dialect - as opposed to the Handshake dialect, which assigns control semantics to _all_ SSA values. As such, data values are
only present in DC where they are required to model control flow.

By having such a control language, the dialect aims to facilitate the construction of dataflow programs where control and data
is explicitly separate. This enables optimization of the control and data side of the program independently, as well as mapping
of the data side into functional units, pipelines, etc.. Furthermore, separating data and control will make it easier to reason
about possible critical paths of either circuit, which may inform buffer placement.

The DC dialect has been heavily influenced by the Handshake dialect, and can either be seen as a successor to it, or a lower level
abstraction. As of writing, DC is _fully deterministic_. This means that non-deterministic operators such as the ones found in Handshake - `handshake.merge, handshake.control_merge` - do **not** have a lowering to DC. Handshake programs must therefore be converted or by construction not contain any of these non-deterministic operators. Apart from that, **all** handshake operations can be lowered to a combination of DC
and e.g. `arith` operations (to represent the data side semantics of any given operation).


## Operations
In DC, there are two main classes of operations:
1. **Control-only operations**: Control-only operations come in pairs and represent start- and end-points in the control network.
   * `fork/join`: A fork splits the incoming token into multiple outgoing tokens - a join is the reverse, wherein it synchronizes all incoming tokens into a single outgoing token.
2. Domain-crossing operations: that is, operations which carry both control- and data semantics.
   * `dc.buffer` is an operation which may buffer an arbitrary amount of data values, and is controlled by a `dc.token` transaction.
   **Hardware note**: buffers have no dialect-side notion of cycles/stages/implementation. It is up to the generating pass to interpret buffer semantics - some may want to add attributes to a single buffer, some may want to stagger `dc.buffer`s sequentially.
   * `branch/select` operations are used to branch control flow into separate baths, and eventually merge them together (`dc.select`). Notably, a branch and select op may have **at most** 2 out/inputs.
   * `pack/unpack` operations are used to pack/unpack values from the control- and data side of the graph. Typically, these are used in cases of data-dependent control flow (branching).


## Canonicalization
By explicitly separating data and control parts of a program, we allow for control-only canonicalization to take place.
Here are some examples of non-trivial canonicalization patterns:
* **Transitive join closure**:
  * Taking e.g. the Handshake dialect as the source abstraction, all operations - unless some specific Handshake operations - will be considered as *unit rate actors* and have join semantics. When lowering Handshake to DC, and by separating the data and control paths, we can easily identify `join` operations which are staggered, and can be merged through a transitive closure of the control graph.
* **Branch to select**: Canonicalizes away a select where its inputs originate from a branch, and both have the same select signal.
* **Identical join**: Canonicalizes away joins where all inputs are the same (i.e. a single join can be used).
