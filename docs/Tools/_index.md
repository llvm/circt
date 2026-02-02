# Tools
CIRCT includes a number of existing tools which can be combined and or extended
to help develop your own custom EDA flows. 

## Front-End Tooling
TODO Overview

## Synthesis Tooling
TODO Overview

## Formal Verification Tooling

Formally verifying hardware designs is a crucial step during the development
process. Various techniques exist, such as logical equivalence checking, model
checking, symbolic execution, etc. The preferred technique depends on the level
of abstraction, the kind of properties to be verified, runtime limitations, etc.
As a hardware compiler collection, CIRCT provides infrastructure to implement
formal verification tooling and already comes with a few tools for common
use-cases. This document provides an introduction to those tools and gives and
overview over the underlying infrastructure for compiler engineers who want to
use CIRCT to implement their custom verification tool.

The
[below diagram](/includes/img/smt_based_formal_verification_infra.svg) provides
a visual overview of the passes and dialects available to support development
of verification tools.

<p align="center"><img src="https://circt.llvm.org/includes/img/smt_based_formal_verification_infra.svg"/></p>

The overall flow will insert an explicit operation to specify the verification
problem (e.g., `verif.lec`, `verif.bmc`). This operation could then be lowered
to an encoding in SMT, an interactive theorem prover, a BDD, or potentially
being exported to existing tools (currently only SMT is supported). Each of
those might have their own different backend paths as well. E.g., an encoding in
SMT can be exported to SMT-LIB or lowered to LLVM IR that calls the Z3 solver.

Existing tools include a [logical equivalence checker](circt-lec.md) and a
[bounded model checker](circt-bmc.md). We discuss extensions to each ofthese
in their respective documentation.
