# Static scheduling infrastructure

Scheduling is a common concern in hardware design, for example in high-level synthesis flows targeting an FSM+Datapath execution model ("static HLS"). This document gives an overview of, and provides rationale for, the infrastructure in the `circt::scheduling` namespace. At its core, it defines an **extensible problem model** that acts as an interface between **clients** (i.e. passes that have a need to schedule a graph-like IR) and reusable **algorithm** implementations.

This infrastructure aims to provide:
- a library of ready-to-use problem definitions and schedulers for clients to hook into.
- an API to make algorithm implementations comparable and reusable.
- a mechanism to extend problem definitions to model additional concerns and constraints.

## Getting started

Let's walk through a simple example. Assume we want to *schedule* the computation in the entry block of a function such as `@foo(...)` in the listing below. This means we want to assign integer *start times* to each of the *operations* in this untimed IR.

```mlir
func @foo(%a1 : i32, %a2 : i32, %a3 : i32, %a4 : i32) -> i32 {
  %0 = arith.addi %a1, %a2 : i32
  %1 = arith.addi %0, %a3 : i32
  %2:3 = "more.results"(%0, %1) : (i32, i32) -> (i32, i32, i32)
  %3 = arith.addi %a4, %2#1 : i32
  %4 = arith.addi %2#0, %2#2 : i32
  %5 = arith.addi %3, %3 : i32
  %6 = "more.operands"(%3, %4, %5) : (i32, i32, i32) -> i32
  return %6 : i32
}
```

Our only constraint is that an operation can start *after* its operands have been computed. The operations in our source IR are unaware of time, so we need to associate them with a suitable *operator type*. Operator types are an abstraction of the target architecture onto which we want to schedule the source IR. Here, the only *property* we need to model is their *latency*. Let's assume that additions take 1 time step, the operations in the dummy `more.` dialect take 3 time steps. As the return operation just passes control back to the caller, we assume a latency of 0 time steps for it.

### Boilerplate

The scheduling infrastructure currently has three toplevel header files.

```c++
//...
#include "circt/Scheduling/Problems.h"
#include "circt/Scheduling/Algorithms.h"
#include "circt/Scheduling/Utilities.h"
//...
using namespace circt::scheduling;
```

### Constructing a problem instance

Our stated goal requires solving an acyclic scheduling problem without resource constraints, represented by the `Problem` class in the scheduling infrastructure. We need to construct an *instance* of the problem, which serves as a container for the problem *components* as well as their properties. The MLIR operation passed as an argument to the `get(...)` method is used to emit diagnostics.

```c++
auto prob = Problem::get(func);
```

Then, we set up the operator types with the latencies as discussed in the introduction. Operator types are identified by string handles.

```c++
auto retOpr = prob.getOrInsertOperatorType("return");
prob.setLatency(retOpr, 0);
auto addOpr = prob.getOrInsertOperatorType("add");
prob.setLatency(addOpr, 1);
auto mcOpr = prob.getOrInsertOperatorType("multicycle");
prob.setLatency(mcOpr, 3);
```

Next, we register all operations that we want to consider in the problem instance, and link them to one of the operator types.

```c++
auto &block = func.getBlocks().front();
for (auto &op : block) {
  prob.insertOperation(&op);
  if (isa<func::ReturnOp>(op))
    prob.setLinkedOperatorType(&op, retOpr);
  else if (isa<arith::AddIOp>(op))
    prob.setLinkedOperatorType(&op, addOpr);
  else
    prob.setLinkedOperatorType(&op, mcOpr);
}
```

Note that we do not have to tell the instance about the *dependences* between the operations because the problem model automatically includes the SSA def-use-edges maintained by MLIR.

### Scheduling

Before we attempt to schedule, we invoke the `check()` method, which ensures that the constructed instance is complete and valid. For example, the check would capture if we had forgot to set an operator type's latency. We dump the instance to visualize the depedence graph.

```c++
auto checkRes = prob.check();
assert(succeeded(checkRes));
dumpAsDOT(prob, "sched-problem.dot");
```

<img src="includes/img/sched-instance.svg"/>

We use a simple list scheduler, available via the `Algorithms.h` header, to compute a solution for the instance.

```c++
auto schedRes = scheduleASAP(prob);
assert(succeeded(schedRes));
```

### Working with the solution

The solution is now stored in the instance, and we invoke the problem's `verify()` method to ensure that the computed start times adhere to the precendence constraint we stated earlier, i.e. operations start after their operands have computed their results. We can also convince ourselves of that by dumping the instance and inspecting the solution.

```c++
auto verifRes = prob.verify();
assert(succeeded(verifRes));
dumpAsDOT(prob, "sched-solution.dot");
```

<img src="includes/img/sched-solution.svg"/>

To inspect the solution programmatically, we can query the instance in the following way. Note that by convention, all getters in the problem classes return `Optional<T>` values, but as we have already verified that the start times for registered operations are set, we can directly dereference the values.

```c++
for (auto &op : prob.getOperations())
  llvm::dbgs() << *prob.getStartTime(&op) << "\n";
```

And that's it! For a more practical example, have a look at the [`AffineToStaticLogic`](https://github.com/llvm/circt/blob/main/lib/Conversion/AffineToStaticLogic/AffineToStaticLogic.cpp) pass.

## Extensible problem model

### Theory and terminology

Scheduling problems come in many flavors and variants in the context of hardware design. In order to make the scheduling infrastructure as modular and flexible as CIRCT itself, it is build on the following idea of an *extensible problem model*:

An *instance* is comprised of *components* called *operations*, *dependences* and *operator types*. Operations and dependences form a graph structure and correspond to the source IR to be scheduled. Operator types encode the characteristics of the target IR. The components as well as the instance can be annotated with *properties*. Properties are either *input* or *solution* properties, based on whether they are supplied by the client, or computed by the algorithm. The values of these properties are subject to the *input constraints* and *solution constraints*, which are a first-class concern in the model and are intended to be strictly enforced before respectively after scheduling.

Concrete problem definitions derived from this model share the same representation of the components, but differ in their sets of properties (and potentially distinction of input and solution properties) and input and solution constraints. Hence, we tie together properties and constraints to model a specific scheduling problem. Extending one (or more!) parent problems means inheriting or adding properties, and redefining the constraints (as these don't always compose automatically).

A key benefit of this approach is that these problem definitions provide a reliable contract between the clients and algorithms, making it clear which information needs to be provided, and what kind of solution is to be expected. Clients can therefore choose a problem definition that fits their needs, and algorithms can *opt-in* to accepting a specific subset of problems, which they can solve efficiently. Extensibility is ensured because new problem definitions can be added to the infrastructure (or inside a specific lowering flow, or even out-of-tree) without adapting any existing users.

### Implementation

See [Problems.h](https://github.com/llvm/circt/blob/main/include/circt/Scheduling/Problems.h) / [Problems.cpp](https://github.com/llvm/circt/blob/main/lib/Scheduling/Problems.cpp).

#### Problem definitions

The `Problem` class is currently the base of the problem hierarchy. Several extended problems are [currently defined](#available-problem-definitions) via virtual multiple inheritance. Upon construction, a `containingOp` is passed to instances. This MLIR operation is currently only used to emit diagnostics, and has no semantic meaning beyond that.

#### Components

The infrastructure uses the following representation of the problem components.

Operations are just `mlir::Operation *`s.

We distinguish two kinds of dependences, *def-use* and *auxiliary*. Def-use dependences are part of the SSA graph maintained by MLIR, and can distinguish specific result and operand numbers. As we expect any relevant graph-like input IR to use this MLIR facility, instances automatically consider these edges between registered operations. Auxiliary dependences, in contrast, only specify a source and destination operation, and have to be explicitly added to the instance by the client, e.g. for control or memory dependences. The `detail::Dependence` class abstracts the differences between both kinds, in order to offer a uniform API to iterate over dependences and query their properties.

Lastly, operator types are identified by `mlir::StringAttr`s, in order to give clients maximum flexibility in modeling their operator library. This may change in the future, when a CIRCT-wide concept to model physical properties of hardware emerges.

#### Properties

Properties can involve arbitrary data types, as long as these can be stored in maps. Problem classes offer public getter and setter methods to access a given components properties. Getters return optional values, in order to indicate if a property is unset. For example, the signature of the method the queries the computed start time is `Optional<unsigned> getStartTime(Operation *op)`. The concrete properties used throughout the problem hierarchy are discussed [below](#available-problem-definitions).

#### Constraints

Clients call the virtual `Problem::check()` method to test any input constraints, and `Problem::verify()` to test the solution constraits. Problem classes are expected to override them as needed. There are no further restrictions of how these methods are implemented, but it is recommended to introduce helper methods that test a specific aspect and can be reused in extended problems. In addition, it makes sense to check/verify the properties in an order that avoids redundant tests for the presence of a particular property as well as redundant iteration over the problem components.

## Available problem definitions

- Problem
- CyclicProblem
- SharedOperatorsProblem
- ModuloProblem
- ChainingProblem

## Available schedulers

- ASAP list scheduler
- Linear programming-based schedulers with integrated simplex solver
- Integer linear programming-based scheduler using external ILP solver

## Utilities

- Topologic graph traversal
- DFA to compute path delays
- DOT dump

## Adding a new problem

- Whereever you want. If it's trait-like and of general use, add to `Problems.h`. Otherwise ok to keep it local.
- Inherit
- Define additional properties, as needed.
   - Redifine `getProperties...(...)` methods to get dumping support
- Redefine `check()` and `verify()`
   - Current organization is having fine-granular `checkXXX/verifyYYY` methods to validate a specific aspect of the problem. These can be reused in subclasses.
- Write tests. See `TestPasses.cpp` for inspiration.

## Adding a new scheduler

- Schedulers should opt-in to specific problems by providing entry points for the problem subclasses they support.
- If schedulers support optimizing for different objectives, they should offer an API for that, as objectives are not part of the problem signature
- Can expect input invariants enforced by `check()` and must compute a solution that complies that passes `verify()`.
- Otherwise, look at the existing implementations for inspiration.
