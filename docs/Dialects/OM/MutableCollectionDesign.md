# Design proposal: append-only `MutableCollection` properties

**Status:** draft for discussion.

## Summary

Introduce an identity-bearing, append-only property collection:

- FIRRTL: `MutableCollection<T>`, represented by `!firrtl.mutable_collection<T>`.
- OM: `!om.collection<T>`.
- Allocation creates an empty collection with fresh identity.
- A resultless push contributes one element to that collection.
- Property/OM programs can transport the handle, but cannot inspect contents.
- The OM **evaluator** executes pushes and publishes completed collections.
  Object elaboration may clone and simplify surrounding IR, but must not execute
  pushes or replace collections with statically assembled lists.

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
2. Reject unresolved external constructor calls that could mutate a collection,
   rather than silently reporting incomplete contents.

## 1. Requirements and scope

`task.md` originally describes `MutableArray`, an allocation resembling
`firrtl.prop.array()`, and a push resembling `firrtl.prop.push(%array, %a)`.
The agreed name for that construct is now `MutableCollection`. Equivalent OM
operations are required, and mutation happens in the evaluator, not the
elaboration transform.

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
- Read-only completed-result access through C++, C, and Python APIs.

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

Runtime collections therefore do not require abandoning elaboration: retain
allocation and push operations through elaboration, then interpret them.

### 2.2 Runtime evaluation is value-oriented

`evaluateObjectInstance` allocates placeholders for all operation results,
evaluates those results, then collects the class fields. `getOrCreateValue`
memoizes by MLIR `Value` in the `objects` map. Runtime kinds currently include
attributes, objects, lists, base paths, and paths.

Important consequences:

- **Resultless pushes are skipped today.** Both operation loops iterate over
  `op.getResults()`. Adding a case to `evaluateValue` cannot execute a push.
- **Expression caching is not effect caching.** Evaluating a collection handle
  must not replay contributions. Each instantiated push needs exactly-once
  execution independently of handle access.
- **Readiness is structural, not necessarily deep.** Elaborated objects install
  references to field placeholders without recursively completing all fields.
  Collection identity and completed membership also need distinct states.

`ListValue::setElements` finishes an immutable list once; it is not an append API.
Use a separate runtime value rather than changing list value semantics.

The current `objects` map belongs to the `Evaluator` and is not cleared by
`instantiate`. Fresh scratch wrappers provide separate IR identities, but new
mutation state should have explicit per-call lifetime instead of relying on this.

### 2.3 Elaboration must preserve residual effects

[`ElaborateObject.cpp`](../../../lib/Dialect/OM/Transforms/ElaborateObject.cpp)
clones constructor bodies, substitutes parameters, produces elaborated objects,
folds field access, and runs greedy rewrites. Strict result verification has a
fixed whitelist in `isFullyEvaluated`; new operations must be accepted as legal
runtime residuals.

`UnknownPropagationPattern` rewrites pure OM operations. Allocation and push
must not be pure, or optimization can erase identity or contributions.

firtool finalization and `om-linker` also run elaboration ahead of the evaluator,
using `all-public-classes` and `allow-unevaluated`. Preservation must work there,
not only in scratch-wrapper instantiation.

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

Collections accept contributions throughout an evaluation transaction and are
complete once all contributions in the instantiated graph are processed.
Completing one object does not complete its collection fields; other objects
may still contribute. Because contributions are executed in a single sweep at
the end of the transaction and construction programs cannot read contents, no
explicit open/sealed state is needed: a collection is only observable outside
the evaluator after its transaction has finished.

Construction programs can allocate, push, transport, and export handles. They
cannot inspect contents or derive computations/assertions from them. No
`collection_get`, `collection_size`, or `collection_to_list` IR operations are
introduced.

Post-evaluation APIs enumerate collections without mutation. Preserve
`!om.collection<T>` as the public type; do not implicitly convert it to a list.

### 3.3 Explicitly unordered

Contributions form an unordered collection with multiplicity. A vector can
store them and a deterministic traversal can provide reproducible output for a
fixed IR, but clients cannot depend on independent pushes' relative positions.

OM classes are graph regions. [MLIR's graph-region contract][graph-regions]
makes operation order within a block semantically insignificant. Unordered
accumulation fits this model without effect tokens, ordering keys, or constructor
sequencing. Memory effects are still needed to preserve contributions.

Use-list order, hash-map iteration, greedy rewrite visitation, and final block
order are not observable ordering guarantees. Resolve handles first, then
collect pushes in any traversal order, exactly once per instantiated push.

### 3.4 Unknown values and external effects

Distinguish unknown elements from unknown membership:

- Pushing an unknown `T` into a known collection contributes one unknown element.
  Known contributions and multiplicity remain visible.
- An unknown collection remains unknown. A push to that destination cannot
  identify a concrete collection and must not affect a known collection.
- Unknown membership is not an empty collection. Enumeration must fail or map
  to the existing Python `Unknown` representation.

This differs intentionally from existing lists, which become unknown when an
element is unknown. Collection unknown status describes membership, not the
presence of an unknown element.

Elaboration currently replaces unresolved external object constructors with
`om.unknown`, discarding parameters. If a constructor could mutate a known
collection parameter, this would silently discard effects.

Initially reject unresolved external constructor calls with collection-bearing
actual parameters, including capabilities reachable through objects, lists,
and opaque inputs. Check before discarding the operands. Account for recursive
types and opaque scratch parameters. If static resolution is insufficient,
preserve the call for evaluator preflight or conservatively reject it.

Primitive-only external classes retain existing behavior. A future effect
summary or retained unknown-contribution operation could instead mark reachable
collections' membership unknown.

## 4. Proposed IR

| Construct | FIRRTL | OM |
| --- | --- | --- |
| Type | `!firrtl.mutable_collection<T>` | `!om.collection<T>` |
| Empty allocation | `firrtl.prop.collection` | `om.collection` |
| Push an element | `firrtl.prop.push` | `om.collection_push` |

Illustrative syntax:

```mlir
%items = firrtl.prop.collection : !firrtl.mutable_collection<string>
firrtl.prop.push %items, %value : !firrtl.mutable_collection<string>, !firrtl.string

%items = om.collection : !om.collection<!om.string>
om.collection_push %items, %value : !om.collection<!om.string>, !om.string
```

FIRRTL `MutableCollectionType` is a non-hardware `PropertyType` parameterized by
a property element type. OM `CollectionType` is distinct from `ListType` and has
an MLIR element type. Lowering recursively converts the element type; never
retain a FIRRTL outer container just because its element already has an OM type.

Verify exact push element/destination element-type agreement. Explicit existing
`AnyRef`/`Any` casts can supply upcasts. Allocation has no operands and one
collection result; push has two operands and no results. Operations are limited
to supported module/class bodies, not hardware `when` regions.

Push consumes a mutation-capable handle; it is not a property-port assignment.
Do not apply property-assignment sink-flow checks to its destination.

Allocation has an allocation effect and is neither pure nor constant-like.
Push has a write effect and is not pure. Start with conservative effects that
do not infer disjoint storage from distinct SSA aliases.

Optimization must not merge allocations, merge identical pushes, remove pushes
because they have no results, or remove unused constructor instances with effects.
Cloning two constructor instances clones contributions twice; accessing one
instance twice does not. Audit instance elimination/duplication, not just pushes.

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
results must not remove the contributions.

## 5. Evaluator design

### 5.1 Runtime value and readiness

Add `EvaluatorValue::Kind::Collection` and `evaluator::CollectionValue`:

```text
CollectionType type
vector<EvaluatorValuePtr> elements
membership-unknown state
```

Provide an internal append method and public read-only enumeration. Update type dispatch, unknown factories, and diagnostic printing.
Append must not be exposed through C/Python bindings.

An allocated handle is immediately structurally ready; `isFullyEvaluated()` is
true while contributions are still pending. This permits transport through
objects/lists and registering an object in a collection referenced by its fields.
There is no separate sealed state: every collection reachable during a
transaction is owned by it (fresh allocations or imported copies), and only the
evaluator can append, so contents are complete whenever a client can see them.

Do not recursively demand completion of the entire reachable graph before
append. Validate structural readiness and use visited sets for whole-graph
checks, since objects can have back-references.

### 5.2 Value phase, then effect phase

The value cache is per-instantiation state. Collections need no additional
transaction-level bookkeeping.

```text
instantiate scratch wrapper:
  1. Prepare transaction-local imported values.
  2. Allocate all result placeholders, including fresh collection handles.
  3. Evaluate ordinary results; establish object fields and handle aliases.
  4. Visit every retained collection_push, regardless of result count.
     Resolve target/element, validate, and append one contribution.
  5. Return the root object (or fail, discarding all transaction-local values).
```

Elaboration puts instantiated bodies in one block, so one effect sweep suffices.
Do not discover effects by traversing public fields: that misses effect-only
instances and may revisit aliased instances.

`getOrCreateValue` allocates a handle once per allocation result.
`evaluateValue` never executes pushes. Add explicit resultless operation dispatch
returning `LogicalResult`. A single sweep enforces exactly-once execution;
if a worklist is later needed, key its execution set by instantiated operations,
not original class-definition operations.

No staging is needed to avoid partial publication: on failure the transaction's
collections are discarded and the caller's inputs were never mutated.
Forward-referenced operands are resolved after placeholders and ordinary values. Diagnose unavailable operands rather than
silently dropping a contribution.

Existing property assertions remain checked by elaboration; there is no need to
move every resultless operation into the interpreter.

### 5.3 API inputs and repeated instantiation

Leaving a collection in the importer's default opaque-argument case would reuse
the caller's original storage and mutate previously returned results.

Proposed policy: **copy on transaction import; alias within the transaction**.

- Accept collections as snapshots with their existing elements.
- Allocate one fresh copy per distinct imported collection identity.
- Use one memoization table across all parameters and nested values.
- Copy/rebind enclosing objects/lists so they reference transaction-local copies
  without modifying the caller's graph.
- Pass runtime collection copies as typed opaque wrapper parameters; no
  collection constant or static-list materialization is needed.
- Handle any-typed inputs consistently: direct and opaque references to the same
  source collection must not receive separate copies.
- Keep existing cyclic concrete-input restrictions and avoid unbounded
  recursive traversal.

Across calls, completed inputs are snapshots. Persistent mutation would need a
separate lifecycle, rollback, and concurrency design.

Per-call cache lifetime does not imply immediate scratch IR deletion:
`ObjectValue` retains a class operation, so existing IR/context lifetime
requirements still apply. Scratch wrapper cleanup is a separate issue.

### 5.4 Elaboration boundary

Accept allocation and push as legal runtime residuals in strict elaboration.
Consider renaming its `isFullyEvaluated` whitelist helper to describe residual
support rather than suggesting effects have executed.

Elaboration clones operations and substitutes operands, folds handle-valued
field access, preserves effects, and checks unsupported external capabilities.
It never appends, gathers contributions from use lists, or replaces collections
with list construction/concatenation. Repeated elaboration must not itself add
runtime elements.

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
| Effect lifetime | Eliminated in FIRRTL `LowerDomains` | Retained through OM elaboration; executed by evaluator |
| Accumulation | Bottom-up list-output plumbing | Shared runtime identity and exactly-once contributions |
| Generated interface | Extra output per registry field | Collection-valued fields/parameters |
| Runtime changes | No new mutable OM value | New value kind, effect phase, and consumer APIs |
| External inputs | Ordinary lowered-list behavior | Explicit capability/unknown-membership policy |

### 7.3 Why the implementations differ

Domain ownership and module relationships let Registry collect contributions
statically and route child outputs upward. Unordered semantics permit regrouping.

A generic collection can be shared through class fields/parameters in an object
graph unrelated to module hierarchy. Identifying the destination instance depends
on actual parameters and aliases. Static removal would require whole-instance
resolution or restricted ownership/alias analysis, rather than copying Registry's
lowering. It also contradicts this task's evaluator-execution requirement.

The PR discussion values removing effects inside FIRRTL
([review comment][registry-review]). The generic facility deliberately trades
that guarantee for composability and reduced hierarchy-specific output plumbing.

Removing the initial Registry type is not a migration of the complete PR.
A future domain Registry frontend could lower to a collection-valued field and
push on its handle, but that changes generated interfaces, input handling,
unknown semantics, and `domaintool`. Evaluate it separately once the generic
mechanism is tested. The reusable concepts are typed append-only construction,
read prohibition, preserved multiplicity, and unordered semantics.

## 8. Consumer APIs

Add separate collection APIs alongside list APIs:

- C++: collection type/identity and read-only enumeration.
- C: type/element-type predicates, runtime-kind predicate, count, and element
  lookup using existing evaluator-value ownership conventions.
- Python: read-only completed-collection length/index/iteration; no append.
  Unknown membership maps to `Unknown`; unknown elements remain visible.

These are external APIs, not property IR reads. Preserve `CollectionType` during
actual-parameter validation, diagnostics, and unknown-value creation. Do not
create an attribute value with a collection type but no collection storage.

## 9. Implementation and validation

Implement in reviewable stages:

1. FIRRTL/OM types and operations, effects, verifiers, and round-trip/error tests;
   remove the initial Registry definition and its obsolete tests.
2. `LowerClasses` conversion, effect-only class discovery, elaboration
   preservation, and path freezing. FIRRTL text syntax can follow separately.
3. Runtime value, resultless-effect sweep, import aliasing, and
   external-capability checks.
4. C/Python consumer APIs and end-to-end tests. Domain Registry integration is
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

Also cover repeated and failed instantiation isolation; imported aliases across
parameters, nested containers, and opaque inputs;
unknown elements versus unknown membership; external-effect diagnostics; repeated
elaboration without execution; operation reordering without changed membership;
per-instance frozen paths; firtool/linker/bytecode preservation; and read-only
consumer APIs.

Assert multiset membership and alias identity, not fixed element order.
Use `unittests/Dialect/OM/Evaluator/EvaluatorTests.cpp`,
`test/Dialect/{OM,FIRRTL}`, `test/CAPI/om.c`, and
`integration_test/Bindings/Python/dialects/om.py`. Run
`ninja -C build check-circt`; enable Python bindings when testing Python APIs.

## 10. Remaining semantic questions

1. Is copy-on-import acceptable across evaluator calls, or is mutation of a
   previously instantiated domain object essential?
2. Can unresolved constructors with collection capabilities be rejected
   initially, or must unknown contributions be represented immediately?
3. Should domain Registry retain pure-list lowering or become a client of the
   generic facility in a later change?

[graph-regions]: https://mlir.llvm.org/docs/LangRef/#graph-regions
[registry-pr]: https://github.com/llvm/circt/pull/10940
[registry-source]: https://github.com/uenoku/circt/tree/c2efe1dc24e15a2d2de050d521f4223fc9f663c5
[registry-review]: https://github.com/llvm/circt/pull/10940#issuecomment-5186428314
