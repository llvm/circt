# Design proposal: append-only `MutableCollection` properties

**Status:** draft for discussion.

## Summary

Introduce an identity-bearing, append-only property collection:

- FIRRTL: `MutableCollection<T>`, represented by `!firrtl.mutable_collection<T>`.
- OM: `!om.collection<T>`.
- Allocation creates a collection with fresh identity, optionally with initial
  elements: `om.collection %a, %b : !om.collection<T>`.
- A resultless push contributes one element to that collection.
- Property/OM programs can transport the handle, but cannot inspect contents.
- Pushes are resolved by **rewriting**: a push into a collection allocated in
  the same class body is folded into the allocation's initial elements. Object
  elaboration inlines constructors and folds field accesses until every push
  reaches its allocation. The evaluator then treats `om.collection` like any
  other value-producing operation and has no effect phase.
- Identity only matters inside one elaboration. At the evaluator API boundary
  a collection is an ordinary list value with unordered elements.

**Confirmed decisions:** the name is `MutableCollection`, and collection order
is unspecified. This proposal preserves multiplicity, like Registry, and needs
no push-sequencing representation.

This generalizes the accumulation use case of [Registry PR #10940][registry-pr]
without relying on domain ownership or bottom-up module output plumbing. The
initial Registry type definition in this checkout can be removed in favor of
this generic property facility. Migrating the full domain Registry prototype is
not part of the initial implementation.

The main semantic decisions still needing agreement are cross-instantiation
inputs and external constructor effects. Recommended initial choices are:

1. Each public `instantiate` call is an isolated evaluation transaction. Imported
   collections are copied into that transaction, preserving aliases within it.
2. Treat unresolved external constructors as today: they become unknown
   objects, and any contributions they might make are not represented.

## 1. Requirements and scope

`task.md` originally describes `MutableArray`, an allocation resembling
`firrtl.prop.array()`, and a push resembling `firrtl.prop.push(%array, %a)`.
The agreed name for that construct is now `MutableCollection`. Equivalent OM
operations are required. Mutation is modeled as a rewrite into the allocation,
which preserves identity; collections are never replaced by lists.

Interpret the prohibition on reads as a restriction on **property/OM
construction programs**. Post-evaluation collateral consumers can enumerate the
completed collection. Handle extraction through a class field is transport, not
a content read.

In scope:

- Collections as property ports, class parameters, and fields.
- Several objects sharing a handle, including through object fields.
- Contributions from objects whose results are otherwise unused.
- Fresh allocations for distinct instances and evaluator calls.
- Duplicate elements, unknown values, and graph-region forward references.
- Completed-result access through the existing list APIs in C++, C, and Python.

Out of scope initially:

- Content indexing, length queries, iteration, membership tests, equality, or
  conversion to a list inside FIRRTL/OM programs.
- Replacement, deletion, deduplication, or clearing.
- Conditional or loop-controlled mutation; pushes reside directly in supported
  module/class bodies.
- Persistent mutable state shared across separate `instantiate` calls.
- Fixing existing cyclic API-input import restrictions or general shared-pointer
  cycle ownership.

## 2. Current evaluator implementation

The implementation is in
[`Evaluator.cpp`](../../../lib/Dialect/OM/Evaluator/Evaluator.cpp), with runtime
values and dispatch declarations in
[`Evaluator.h`](../../../include/circt/Dialect/OM/Evaluator/Evaluator.h).

### 2.1 Instantiation includes IR elaboration

```text
Evaluator::instantiate(class, actual parameters)
  verifyActualParameters
  ScratchIRBuilder::run
    create a temporary wrapper class
    import API inputs as IR / opaque wrapper parameters
    create an om.object for the requested class
    run om-elaborate-object on the wrapper
  instantiateImpl(wrapper, opaque parameters)
    evaluateObjectInstance
  return wrapper.root
```

The scratch importer materializes primitives as constants, lists as
`om.list_create`, and concrete objects as `om.elaborated_object`. Any-typed inputs
are opaque wrapper parameters. It memoizes imported runtime values and rejects
mutually referential concrete object inputs.

Elaboration already inlines every constructor into the wrapper class, so every
push ends up in the same block as its allocation and can be folded into it.

### 2.2 Runtime evaluation is value-oriented

`evaluateObjectInstance` allocates placeholders for all operation results,
evaluates those results, then collects the class fields. `getOrCreateValue`
memoizes by MLIR `Value` in the `objects` map. Runtime kinds currently include
attributes, objects, lists, base paths, and paths.

Important consequences:

- **Resultless pushes are skipped.** Both operation loops iterate over
  `op.getResults()`. Rather than adding an effect phase, pushes are folded into
  their allocation before evaluation, so contents are an ordinary operand list.
- **Readiness is structural, not necessarily deep.** Elaborated objects install
  references to field placeholders without recursively completing all fields.
  Collections do the same for their elements, which permits back-references.

`ListValue::setElements` finishes an immutable list once; it is not an append API.
Use a separate runtime value rather than changing list value semantics.

The current `objects` map belongs to the `Evaluator` and is not cleared by
`instantiate`. Fresh scratch wrappers provide separate IR identities, but new
mutation state should have explicit per-call lifetime instead of relying on this.

### 2.3 Elaboration resolves effects

[`ElaborateObject.cpp`](../../../lib/Dialect/OM/Transforms/ElaborateObject.cpp)
clones constructor bodies, substitutes parameters, produces elaborated objects,
folds field access, and runs greedy rewrites. Strict result verification has a
fixed whitelist of legal residual operations. Adding the push canonicalization
to its pattern set folds contributions into allocations as soon as inlining
exposes them. Pushes remain only when they target a class parameter.

`UnknownPropagationPattern` rewrites pure OM operations. Allocation and push
must not be pure, or optimization can erase identity or contributions; in
particular an unknown initial element must not make the collection unknown.

firtool finalization and `om-linker` also run elaboration ahead of the evaluator,
using `all-public-classes` and `allow-unevaluated`. Folding is valid there too,
since it never crosses a class boundary.

## 3. Semantic contract

### 3.1 Identity and aliasing

Each instantiated allocation creates a fresh collection. Equal types, contents,
or constructor arguments do not imply identical collections.

Property assignment and parameter/field transport pass the **handle**, not a
copy of the contents. Existing single-assignment rules govern property bindings;
push mutates referenced storage, not the binding.

Two children receiving the same parent handle contribute to one collection.
Children allocating their own collections remain distinct. Two distinct pushes
of the same value contribute two elements. Accessing an object through multiple
aliases must not replay its constructor effects.

### 3.2 Construction and observation

A collection's contents are the contributions of the entire instantiated graph,
not just of the object that allocates it; other objects may contribute through
shared handles. Because construction programs cannot read contents, folding
pushes into the allocation is semantics-preserving, and no explicit
open/sealed state is needed: once elaboration has folded every push, the
allocation lists the complete membership.

Construction programs can allocate, push, transport, and export handles. They
cannot inspect contents or derive computations/assertions from them. No
`collection_get`, `collection_size`, or `collection_to_list` IR operations are
introduced.

Post-evaluation, a collection is exposed as a list value whose element order is
unspecified. In IR, `!om.collection<T>` remains distinct from `!om.list<T>`.

### 3.3 Explicitly unordered

Contributions form an unordered collection with multiplicity. A vector can
store them and a deterministic traversal can provide reproducible output for a
fixed IR, but clients cannot depend on independent pushes' relative positions.

OM classes are graph regions. [MLIR's graph-region contract][graph-regions]
makes operation order within a block semantically insignificant. Unordered
accumulation fits this model without effect tokens, ordering keys, or constructor
sequencing. Memory effects are still needed to preserve contributions.

Use-list order, hash-map iteration, greedy rewrite visitation, and final block
order are not observable ordering guarantees. Folding may therefore gather
pushes in any order, as long as each instantiated push is folded exactly once.

### 3.4 Unknown values and external effects

Distinguish unknown elements from unknown membership:

- Pushing an unknown `T` into a known collection contributes one unknown element.
  Known contributions and multiplicity remain visible.
- An unknown collection remains unknown. A push to that destination cannot
  identify a concrete collection and must not affect a known collection; it is
  dropped by canonicalization.
- Unknown membership is not an empty collection. Enumeration must fail or map
  to the existing Python `Unknown` representation.

This differs intentionally from existing lists, which become unknown when an
element is unknown. Collection unknown status describes membership, not the
presence of an unknown element.

Elaboration replaces unresolved external object constructors with `om.unknown`,
discarding parameters. Collections passed to such a constructor keep their
known contributions; whatever the external class would contribute is not
represented. This keeps external classes uniform with existing behavior and
avoids a capability analysis over objects, lists, and opaque inputs. Linking
external classes before evaluation (`om-linker`) resolves the common case. A
future effect summary or retained unknown-contribution operation could instead
mark reachable collections' membership unknown.

## 4. Proposed IR

| Construct | FIRRTL | OM |
| --- | --- | --- |
| Type | `!firrtl.mutable_collection<T>` | `!om.collection<T>` |
| Allocation | `firrtl.prop.collection` | `om.collection` |
| Push an element | `firrtl.prop.push` | `om.collection_push` |

Illustrative syntax:

```mlir
%items = firrtl.prop.collection : !firrtl.mutable_collection<string>
firrtl.prop.push %items, %value : !firrtl.mutable_collection<string>, !firrtl.string

%items = om.collection : !om.collection<!om.string>
om.collection_push %items, %value : !om.collection<!om.string>, !om.string

// Canonical form of the two OM operations above.
%items = om.collection %value : !om.collection<!om.string>
```

FIRRTL `MutableCollectionType` is a non-hardware `PropertyType` parameterized by
a property element type. OM `CollectionType` is distinct from `ListType` and has
an MLIR element type. Lowering recursively converts the element type; never
retain a FIRRTL outer container just because its element already has an OM type.

Verify exact push element/destination element-type agreement, and likewise for
initial elements. Explicit existing `AnyRef`/`Any` casts can supply upcasts.
OM allocation takes variadic initial elements and has one collection result;
push has two operands and no results. Operations are limited
to supported module/class bodies, not hardware `when` regions.

Push consumes a mutation-capable handle; it is not a property-port assignment.
Its collection operand is read like any other property source, so it must have
source or duplex flow: push through a local wire or an input, not through an
output port. This keeps `LowerClasses` free of special output-port handling.

Allocation has an allocation effect and is neither pure nor constant-like.
Push has a write effect and is not pure. Start with conservative effects that
do not infer disjoint storage from distinct SSA aliases.

Optimization must not merge allocations, merge identical pushes, remove pushes
because they have no results, or remove unused constructor instances with effects.
Cloning two constructor instances clones contributions twice; accessing one
instance twice does not. Audit instance elimination/duplication, not just pushes.
An unused allocation may be removed, together with pushes already folded into
it, since its contents are unobservable.

### Example: contributions through an object field

```mlir
!items = !om.collection<!om.integer>

om.class @Owner() -> (items: !items) {
  %items = om.collection : !items
  om.class.fields %items : !items
}

om.class @Writer(%items: !items, %value: !om.integer) {
  om.collection_push %items, %value : !items, !om.integer
  om.class.fields
}

om.class @Top() -> (owner: !om.class.type<@Owner>, items: !items) {
  %owner = om.object @Owner() : () -> !om.class.type<@Owner>
  %items = om.object.field %owner["items"] :
      (!om.class.type<@Owner>) -> !items
  %one = om.constant #om.integer<1 : si8> : !om.integer
  %two = om.constant #om.integer<2 : si8> : !om.integer
  %a = om.object @Writer(%items, %one) :
      (!items, !om.integer) -> !om.class.type<@Writer>
  %b = om.object @Writer(%items, %two) :
      (!items, !om.integer) -> !om.class.type<@Writer>
  om.class.fields %owner, %items : !om.class.type<@Owner>, !items
}
```

`Top.items` and `Top.owner.items` are the same runtime collection, containing
one contribution of `1` and one of `2`, in unspecified order. Unused writer
results must not remove the contributions. After elaboration, `Top` contains
`om.collection %one, %two : !items` and no push.

## 5. Evaluator design

### 5.1 Runtime value and readiness

No new runtime value kind is needed. Once elaboration has folded every push, a
collection is fully described by its allocation's operands, and nothing can
observe its identity across `instantiate` calls (inputs are copied). The
evaluator therefore produces a `ListValue` of `!om.list<T>` for each
`om.collection` of `!om.collection<T>`.

`getOrCreateValue` allocates a placeholder list once per `om.collection`
result; `evaluateValue` sets its elements from the operand handles, without
forcing their evaluation, as `om.elaborated_object` does for fields. This
permits registering an object in a collection referenced by its own fields.
Unlike `om.list_create`, an unknown element does not make the collection
unknown; an `om.unknown` collection is an unknown list.

### 5.2 No effect phase

```text
instantiate scratch wrapper:
  1. Import API inputs as IR, including fresh allocations for collections.
  2. Elaborate; every push is folded into its allocation.
  3. Allocate all result placeholders and evaluate them as usual.
  4. Return the root object.
```

Elaboration puts instantiated bodies in one block, and the wrapper never has
collection-typed parameters, so no push survives elaboration and the evaluator
needs no knowledge of pushes. Exactly-once contribution is a property of the
rewrite: each instantiated push becomes one operand. Failure discards the
scratch IR and its values, so partial results are never published.

### 5.3 API inputs and repeated instantiation

Inputs follow value semantics: a list passed where a collection is expected
supplies its **initial contents**. This is why a collection passed to
`instantiate` is never mutated by the call, and why exposing it as a list is
sufficient.

- Accept a list value for a collection parameter with the same element type.
- Materialize every collection-typed use of a list, whether a parameter or a
  nested object field/list element, as a fresh `om.collection` whose operands
  are the imported elements. These allocations are not memoized, so two uses
  of the same input list are independent collections, exactly as if two list
  values had been passed. The import memoization table is unchanged.
- Concrete objects/lists are already materialized as IR, so they reference the
  copies without modifying the caller's graph.
- Any-typed inputs stay opaque, as for every other value. Since OM programs
  cannot downcast `!om.any`, they cannot push through an opaque reference, so
  they keep referring to the caller's (unmodified) original.
- Keep existing cyclic object-input restrictions. A collection containing an
  object that refers back to it is such a cycle and is rejected on import.

Across calls, completed inputs are snapshots. Persistent mutation would need a
separate lifecycle, rollback, and concurrency design.

Per-call cache lifetime does not imply immediate scratch IR deletion:
`ObjectValue` retains a class operation, so existing IR/context lifetime
requirements still apply. Scratch wrapper cleanup is a separate issue.

### 5.4 Elaboration boundary

Elaboration clones operations and substitutes operands, folds handle-valued
field access, and folds pushes into allocations. It never replaces collections with list construction or
concatenation. Allocations and pushes into class parameters are legal residuals.

## 6. FIRRTL lowering and paths

[`LowerClasses.cpp`](../../../lib/Dialect/FIRRTL/Transforms/LowerClasses.cpp)
already moves operations with property operands/results into OM class bodies.
A correctly classified resultless push is recognized through its operands.

```text
MutableCollection<T>     -> Collection<convert(T)>
firrtl.prop.collection   -> om.collection
firrtl.prop.push(a, v)    -> om.collection_push(convert(a), convert(v))
```

Preserve handle assignment rather than copying elements. Property discovery in
`lib/Analysis/FIRRTLInstanceInfo.cpp` scans operands and results, so effect-only
usage should generate an OM class; add a regression test. Audit visitors,
property classification, parser/emitter support, and unused-instance pruning.

FIRRTL textual type/allocation/push syntax needs feature-version agreement.
MLIR operations and lowering can be implemented first without assigning a
speculative FIRRTL language version.

[`FreezePaths.cpp`](../../../lib/Dialect/OM/Transforms/FreezePaths.cpp)
recursively replaces path types and updates arguments/fields, but explicitly
recreates only selected operations such as list construction. Update collection
allocation result types too, including nested containers, and verify push
operands agree after freezing. Preserve per-instance base paths and operation
attributes. Do not freeze by flattening collections into lists.

## 7. Comparison with Registry PR #10940

### 7.1 Reviewed prototype

This comparison uses the PR description, discussion, and diff at head
`c2efe1dc24e15a2d2de050d521f4223fc9f663c5` ([source snapshot][registry-source]).
It describes a prototype, not accepted upstream semantics.

At the start of this work, the checkout included an initial `RegistryType` and
parser/emitter tests, but not the full insertion/lowering implementation.
The initial definition is removed by the collection implementation. The reviewed
PR itself moves Registry out of `PropertyType` and restricts it to domain fields.
It is not an existing general mutable property handle.

In that prototype:

- Domain fields implicitly own initially empty registries and exclude them from
  ordinary domain constructor inputs.
- `firrtl.domain.insert` targets a domain and field name. Its property-typed
  source is checked against the element type, despite path-focused examples.
- Inserts are restricted to module bodies. `LowerDomains` collects local inserts
  associated with module domain ports and removes the insertion operations.
- Each registry becomes a separate bottom-up list output; parents concatenate
  local contributions and associated child-instance registry lists.
- OM receives ordinary lists, not mutable operations.
- `domaintool` associates sibling outputs such as `A_registry_clockGates` with
  domains and emits hierarchical path strings.
- Contents are unordered and not deduplicated. Concrete output order is a
  representation choice, not an insertion-order guarantee.

### 7.2 Design points

| Design point | Registry prototype | `MutableCollection` |
| --- | --- | --- |
| Purpose | Domain asset tracking | General property/OM accumulation |
| Ownership | Implicit domain-field ownership | Explicit allocation; shared handles |
| Type category | Domain-only, outside `PropertyType` in reviewed PR | Ordinary property type plus OM container type |
| Mutation target | Domain identity plus field name | First-class collection handle |
| Allocation | Implicit empty registry | Fresh allocation per instantiated operation |
| Element type | Property-typed; examples emphasize paths | Supported property/OM element types |
| Program reads | Content reads disallowed | Content reads disallowed; handle transport allowed |
| Order/multiplicity | Unordered; duplicates preserved | Unordered; duplicates preserved |
| Effect lifetime | Eliminated in FIRRTL `LowerDomains` | Folded into the allocation during OM elaboration |
| Accumulation | Bottom-up list-output plumbing | Shared runtime identity and exactly-once contributions |
| Generated interface | Extra output per registry field | Collection-valued fields/parameters |
| Runtime changes | No new mutable OM value | None; evaluated collections are lists |
| External inputs | Ordinary lowered-list behavior | Ordinary unknown-object behavior |

### 7.3 Why the implementations differ

Domain ownership and module relationships let Registry collect contributions
statically and route child outputs upward. Unordered semantics permit regrouping.

A generic collection can be shared through class fields/parameters in an object
graph unrelated to module hierarchy. Identifying the destination instance depends
on actual parameters and aliases. Static removal would require whole-instance
resolution or restricted ownership/alias analysis, rather than copying Registry's
lowering. Folding within the elaborated OM object graph avoids that analysis.

The PR discussion values removing effects inside FIRRTL
([review comment][registry-review]). The generic facility deliberately trades
that guarantee for composability and reduced hierarchy-specific output plumbing.

Removing the initial Registry type is not a migration of the complete PR. The
reusable concepts are typed append-only construction, read prohibition,
preserved multiplicity, and unordered semantics.

### 7.4 Example: a clock-gate registry on a domain

A domain field can already be a `MutableCollection`, so the Registry use case
needs no domain-specific mechanism. See
[`domain-collections.fir`](../../../test/firtool/domain-collections.fir):

```firrtl
domain ClockDomain:
  name: String
  clockGates: MutableCollection<Path>

module ClockGate:
  input D: Domain of ClockDomain
  input clock: Clock domains [D]
  output gated: Clock domains [D]
  connect gated, clock
  push D.clockGates, path("OMReferenceTarget:~Top|ClockGate>gated")

public module Top:
  input A: Domain of ClockDomain
  ; ...
  inst g0 of ClockGate
  domain_define g0.D = A
  inst g1 of ClockGate
  domain_define g1.D = A
```

`LowerDomains` turns the domain into an `@ClockDomain` class with a
`clockGates` collection field, and `D.clockGates` into a field access on the
domain object. firtool's elaboration inlines both gates into `Top_Class`, each
pushing a path under its own base path (`Top/g0`, `Top/g1`). No registry output
ports or parent concatenation are generated.

The pushes target the domain parameter, so they remain in `Top_Class` until
evaluation. A consumer creates the domain with an empty list and passes it in:

```text
A   = instantiate ClockDomain(basepath, "A", [])
top = instantiate Top_Class(basepath, A)
top.A_out.domainInfo_out.clockGates_out
  == [~Top|Top/g0:ClockGate>gated, ~Top|Top/g1:ClockGate>gated]   (unordered)
```

The input `A` itself is unchanged; the populated domain is reported through the
existing `ClockDomain_out` object (unit test `CollectionDomainRegistry`).
Remaining follow-up: `domaintool` only passes string/integer domain arguments,
so it needs to pass an empty list for collection fields and to read the
collection back when generating collateral.

## 8. Consumer APIs

Evaluated collections use the existing list APIs in C++, C, and Python.
Unknown membership maps to an unknown list (`Unknown` in Python); unknown
elements remain visible. Only the IR type gets new C/Python bindings
(`CollectionType` and its element type). Unknown values of collection type are
created as lists.

## 9. Implementation and validation

Implement in reviewable stages:

1. FIRRTL/OM types and operations, effects, verifiers, and round-trip/error tests;
   remove the initial Registry definition and its obsolete tests.
2. `LowerClasses` conversion, effect-only class discovery, elaboration
   preservation, and path freezing. FIRRTL text syntax can follow separately.
3. Initial-element allocation, push folding, runtime value, import aliasing,
   and the FIRRTL push flow check.
4. C/Python type bindings and end-to-end tests. Domain Registry integration is
   independent follow-up work.

Principal files:

| Area | Files |
| --- | --- |
| FIRRTL types/ops | `include/circt/Dialect/FIRRTL/FIRRTLTypes{,Impl}.td`, `FIRRTLTypes.h`, `FIRRTLExpressions.td`, `FIRRTLStatements.td`, `FIRRTLVisitors.h`; `lib/Dialect/FIRRTL/FIRRTLTypes.cpp`, `FIRRTLOps.cpp` |
| FIRRTL text | `lib/Dialect/FIRRTL/Import/FIRParser.cpp`, `FIRTokenKinds.def`; `lib/Dialect/FIRRTL/Export/FIREmitter.cpp` |
| Lowering/discovery | `lib/Dialect/FIRRTL/Transforms/LowerClasses.cpp`; `lib/Analysis/FIRRTLInstanceInfo.cpp` |
| OM types/ops | `include/circt/Dialect/OM/OMTypes.td`, `OMOps.td`; `lib/Dialect/OM/OMOps.cpp` |
| OM preparation | `lib/Dialect/OM/Transforms/ElaborateObject.cpp`, `FreezePaths.cpp`; `include/circt/Dialect/OM/OMPasses.td` |
| Runtime | `include/circt/Dialect/OM/Evaluator/Evaluator.h`; `lib/Dialect/OM/Evaluator/Evaluator.cpp` |
| Bindings | `include/circt-c/Dialect/OM.h`; `lib/CAPI/Dialect/OM.cpp`; `lib/Bindings/Python/OMModule.cpp` |

Tests must cover empty and nonempty collections; duplicate pushes; distinct
allocations under CSE; shared versus independent child handles; unused writers;
field transport and forward references; repeated object access without replay;
and an object registered in a collection referenced by its own fields.

Also cover repeated and failed instantiation isolation; independent copies of
imported lists across parameters and nested containers; cyclic-input rejection;
unknown elements versus unknown membership; push
folding and pushes into parameters; operation reordering without changed membership;
per-instance frozen paths; firtool/linker/bytecode preservation; and list
inputs at collection parameters.

Assert multiset membership and alias identity, not fixed element order.
Use `unittests/Dialect/OM/Evaluator/EvaluatorTests.cpp`,
`test/Dialect/{OM,FIRRTL}`, `test/CAPI/om.c`, and
`integration_test/Bindings/Python/dialects/om.py`. Run
`ninja -C build check-circt`; enable Python bindings when testing Python APIs.

## 10. Remaining semantic questions

1. Is copy-on-import acceptable across evaluator calls, or is mutation of a
   previously instantiated domain object essential?
2. Is ignoring contributions from unresolved external constructors acceptable,
   or must unknown contributions be represented?
3. Should domain Registry retain pure-list lowering or become a client of the
   generic facility in a later change?

[graph-regions]: https://mlir.llvm.org/docs/LangRef/#graph-regions
[registry-pr]: https://github.com/llvm/circt/pull/10940
[registry-source]: https://github.com/uenoku/circt/tree/c2efe1dc24e15a2d2de050d521f4223fc9f663c5
[registry-review]: https://github.com/llvm/circt/pull/10940#issuecomment-5186428314
