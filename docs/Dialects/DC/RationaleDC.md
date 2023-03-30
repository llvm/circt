# DC Dialect Rationale

[TOC]

## Introduction

The intention of DC is to model all operations required to represent control flow as a graph of SSA values.
In doing so, we separate the notion of the functional/data parts of a given program with the control-flow aspects,
as well as have clear demarkations for when one influences the other (e.g. conditional branching).

## Operations
In `dc`, there are two main classes of operations:
1. **Control-only operations**: Control-only operations come in pairs and represent start- and end-points in the control network.
  * `fork/join`: A fork splits the incoming token into multiple outgoing tokens - a join is the reverse, wherein it synchronizes all incoming tokens into a single outgoing token.
2. Domain-crossing operations:
   * `dc.buffer` is an operation which may buffer an arbitrary amount of data values, and is controlled by a `dc.token` transaction.
   **Hardware note**: buffers have no dialect-side notion of cycles/stages/implementation. It is up to the generating pass to interpret buffer semantics - some may want to add attributes to a single buffer, some may want to stagger `dc.buffer`s sequentially.
   * `branch/merge`: Notably, a branch and merge op may have **at most** 2 out/inputs. In `dc`, determism is assumed, and by extension, a `merge` will always *eventually* be driven by the input of a corresponding `branch` operation. A `branch` operation is expected to be driven by a data-side boolean value.
   * `pack/unpack` operations are used to pack/unpack values from the control- and data side of the graph. Most typically, these are used in cases of data-dependent control flow (branching).
   * `dc.node, dc.data, dc.control`: These are symbol defining/referencing operations which establish a mapping between the control and data side of the graph. `dc.data` and `dc.control` have no functional semantics, and simply serves as a means to associating a symbol with a def/use chain in the graph.

## Modeling control- and data graph interactions
When analyzing and transforming the `dc` dialect, we expect (based on the output abstraction) that there will often be a need to recursively analyze and optimize either the data- or control side of the graph, whereafter modifications must be performed in each side of the graph to maintain function semantics.

 <img title="DC symbols" src="includes/img/dc-symbols.png"/>

To facilitate this, `dc` has the capability to describe symbols wherein anchors in the control- and data network, respectively, can be placed and associate these symbols. These symbols are intended to be initially assigned upon IR construction. `dc` then has the capability track control- and data graph semantic "locations" through canonicalization:
* Move symbols upwards in the SSA def-use chain.
* Merge symbols when these coincide in either the control- or data part of the graph.
  * E.g. control/data-side canonicalization may have resulted in operations being removed which thus required merging symbol referenced at a given point in the graph. This means that there can be a 1:N association between locations in either side of the graph.

```mlir
hw.module @roundtrip(
        %sel : !dc.value<i1>,
        %a: !dc.value<i64>,
        %b : !dc.value<i64>,
        %c : !dc.value<i64>) -> (!dc.value<i64>) {

    %sel_t, %sel_v = dc.unwrap %sel : !dc.value<i1>
    %a_t, %a_v = dc.unwrap %a : !dc.value<i64>
    %b_t, %b_v = dc.unwrap %b : !dc.value<i64>
    %c_t, %c_v = dc.unwrap %c : !dc.value<i64>

    // Symbols
    dc.symbol @nMux
    dc.symbol @nAdd

    // Control side
    %0 = dc.merge %sel, %a_t, %b_t : !dc.token
    %1 = dc.control @nMux %0
    %2 = dc.join %0, %c_t : !dc.token
    %3 = dc.control @nAdd %2

    // Data side
    %4 = arith.select %sel_v, %a_v, %b_v : i64
    %5 = dc.data @nMux %4 : i64
    %3 = arith.addi %2, %c_v : i64
    %6 = dc.data @nAdd %3 : i64

    // Wrap up
    %7 = dc.wrap %3, %6 : !dc.token, i64

    hw.output %7 : !dc.value<i64>
}
```

The above capabilities thus gives us a method to recursively perform the following:
1. Canonicalize the control and data side of the graph independently
2. Update symbols and symbol references to maintain inter-graph semantics
3. Perform analysis of either the control and data side of the graph and transform the circuit while maintaining semantic equivalence
   * As an example, say we aim to lower `dc` IR into a hardware representation, we may want to perform critical path estimation or buffer placement of either the control or data side of the graph. Importantly, we want to do this *after* canonicalizing the respective sides of the graph. Given the presence of the inter-graph symbols, we are always able to know where to place buffers to ensure correctness.

## Canonicalization
By performing this separation, we allow for the IR to be easily analyzed, transformed and optimized.
Example canonicalizations:
* **Transitive join closure**:
  * Taking e.g. the `handshake` dialect as the source abstraction, all operations - unless some specific `handshake` operations - will be considered as *unit rate actors* and have join semantics. When lowering `handshake` to `DC`, and by separating the data and control paths, we can easily identify `join` operations which are staggered, and can be merged through a transitive closure of the control graph.


## TODO:

Ideally `dc` operations should go into a `func.func` since i don't really feel like making another top-level function kind (`dc.func`)... the only thing we really need from the containing operation is for it to be a graph region. And i don't want to restrict it to live within `hw.module`.
