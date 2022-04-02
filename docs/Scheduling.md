# Static scheduling infrastructure

- Intro, target audience

## Getting started

- Example
- Problem construction
- Check, verify
- Inspect solution
- See AffineToStaticLogic for this all in action

## Extensible problem model

- Instance and its components: Operations, dependences and operator types. Implicit SSA dependences vs. explicit "auxiliary" dependences.
- Properties
- Input and solution constraints
- Rationale for having a hierarchy of problems instead of an all-encompassing model
- Rationale for the KV-API instead of tying it more directly to a dialect etc.

## Available problem definitions

- Problem
- CyclicProblem
- SharedOperatorsProblem
- ModuloProblem
- ChainingProblem

## Available schedulers

- ASAP list scheduler
- Linear programming-based schedulers with integrated simplex solver
- Integer linear programming-based scheduer using external ILP solver

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
