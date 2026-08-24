# Hierarchy Manipulation Rationale

This document explains the rationale behind `circt-hm`: a tool for restructuring
the module/instance hierarchy of a hardware design, without changing its
behavior.

## Motivation

### Physical Design

The driving use case is physical design (PD) and floorplanning. Designers
decompose a chip logically (along team, feature, and reuse boundaries) but PD
wants the hierarchy carved into units that reflect physical locality and timing,
not the logical decomposition. Without hierarchy manipulation, we would be
forced to pollute the logical description of the circuit with physical concerns.

Hierarchy manipulation lets a designer describe the reorganization declaratively
and apply it as a structural rewrite. The essential constraint is that these
rewrites **preserve the behavior of the design**: reorganizing the hierarchy
must not change what the circuit computes, only how its modules and instances
are grouped.

We get this guarantee from the shape of the rules themselves. A rewrite rule is
explicit about what must be present in the design for it to apply, and about the
structure it produces. It matches a left-hand-side (LHS) pattern of modules and
instances, and replaces it with a right-hand-side (RHS) replacement. The
compiler verifies that the two are behaviorally equivalent, that every module
instance in the pattern is preserved by the rewrite, before any mutation is
attempted. Because the rules name exactly what they depend on, a rule that would
silently drop or reconnect a signal simply fails to verify, rather than
producing a design that no longer behaves the same. We can verify a rewrite
independently of the design it is applied to.

### Replace "hierarchy manipulation" Compiler Passes

A number of transformations in the `firtool` compiler are, at their core,
hierarchy manipulations. They are currently implemented as ad-hoc C++ passes,
each manually walking the instance graph, updating ports, and rewiring dataflow,
and each driven by its own combination of FIRRTL annotations and command-line
flags. Concrete examples include:

- inject DUT hierarchy: wraps the design under test (DUT) under a new parent
  module.
- extract instances: relocates black-boxes, clock gates, and memories.
- module inliner: user-directed module instance inliner.

These passes reimplement the same fragile hierarchy-surgery logic in slightly
different ways. Expressing them instead as hierarchy manipulation rules would
replace that bespoke, dialect-specific code with declarative rules that reuse a
single, verified rewriting engine, and would let the annotations and flags that
drive them be expressed as hierarchy manipulation scripts instead.

### A Generic Tool for Hierarchy Manipulation

While simplifying firtool is a concrete goal, we aim to build a tool general
enough to serve hierarchy manipulation needs across a variety of workflows and
MLIR dialects. The same engine and rule language should apply whether a design
is expressed in `firrtl`, `hw`, or any other dialect with a module hierarchy.

## Design

### Virtual Machine (VM) with Backtracking

The pass-replacement use cases above are what push us past simple,
statically-known rewrites. A rule like "extract every memory" cannot be a single
fixed pattern: the compiler does not know up front how many memories exist,
where they sit, or which surrounding structure each one needs. These are
**dynamic rules**: rules the compiler attempts repeatedly, guided by a search
over the design, where any individual attempt may fail because its preconditions
are not met.

Search with failure means speculation, and speculation means we need to undo
work cheaply. Mutating the MLIR IR directly and then trying to roll back IR
edits on a failed attempt is slow and fragile. So rewrites do not run against
the IR directly. Instead, rules compile to bytecode that executes on a virtual
machine (VM) operating over a lightweight, cache-friendly representation of the
instance graph. The VM is built around backtracking: an attempt runs as a
transaction, and if a constraint fails partway through, the VM unwinds to the
last decision point and tries an alternative, all without ever touching (or
having to repair) the real IR. This makes "try rule A, and if it does not apply,
try rule B" the natural mode of execution rather than a special case, which is
exactly what a search over dynamic rules requires.

### Dialect-agnostic

Hierarchy is a concept shared by many of our IRs: `hw.instance`,
`firrtl.instance`, and others. We do not want a separate rewriting engine per
dialect. The framework is therefore deliberately agnostic about which IR it is
ultimately rewriting. The VM operates only on its own abstract graph and knows
nothing about MLIR operations, types, or dataflow. Bridging the VM to a concrete
dialect takes two pieces per IR: an importer on the way in and a replayer on the
way out.

The **importer** builds the VM's initial instance graph from the design. It
walks the dialect's IR and produces the abstract nodes and edges (modules and
instances) that the VM matches and rewrites against. This is the only point at
which the framework reads dialect-specific structure; from here on the VM sees
nothing but its own graph.

The bridge on the way back out is a **journal**. As the VM executes (and
backtracks), it records the sequence of structural actions that make up the
committed rewrite: nodes and edges created, deleted, and reconnected. This
journal is the sole output of running a rule, and it is completely
IR-independent.

Applying a rewrite to the real IR is then a matter of *replaying* the journal.
For each dialect we implement a **replayer** that consumes the journal and
performs the corresponding mutations on that dialect's IR: creating the
`hw.instance` or `firrtl.instance` operations, moving modules, and rewiring
ports. Crucially, replay is not a failable process and never needs to backtrack.
All of the searching, speculation, and failure happened inside the VM; the
journal contains only the actions that survived backtracking and describes a
rewrite already known to be valid. The replayer therefore just faithfully
reproduces that result, never reasoning about the search itself. Adding support
for a new IR means writing an importer and a replayer, not a new rewriting
engine.
