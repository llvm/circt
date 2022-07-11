# SSP Dialect Rationale

This document describes various design points of the SSP dialect, why they are
the way they are, and current status.  This follows in the spirit of other
[MLIR Rationale docs](https://mlir.llvm.org/docs/Rationale/).

## Introduction

CIRCT's [scheduling infrastructure](https://circt.llvm.org/docs/Scheduling/) is lightweight and dialect-agnostic, in order to fit into any lowering flow with a need for static scheduling. However, it lacks an import/export format for storing and exchanging problem instances. The SSP ("**S**tatic **S**cheduling **P**roblems") dialect fills that role by defining an IR that captures problem instances 
- in full fidelity,
- in a concise syntax,
- and independent of any other "host" dialect.

The SSP dialect's main use-cases are [testing](#testing), [benchmarking](#benchmarking) and [rapid-prototyping](#rapid-prototyping). It is strictly a companion to the existing scheduling infrastructure, and clients (HLS flows etc.) are advised to use the C++ APIs directly.

### Testing

In order to test the scheduling infrastructure's problem definitions (in particular, their input checkers/solution verifiers) and algorithm implementations, a "host" IR to store problem instances is needed. To that end, the test-cases started out with a mix of standard and unregistered operations, and heavy use of generic attributes, as shown in the following example (note especially the index-based specification of auxiliary dependences):

```mlir
func.func @canis14_fig2() attributes {
  problemInitiationInterval = 3,
  auxdeps = [ [3,0,1], [3,4] ],
  operatortypes = [
    { name = "mem_port", latency = 1, limit = 1 },
    { name = "add", latency = 1 }
  ] } {
  %0 = "dummy.load_A"() { opr = "mem_port", problemStartTime = 2 } : () -> i32
  %1 = "dummy.load_B"() { opr = "mem_port", problemStartTime = 0 } : () -> i32
  %2 = arith.addi %0, %1 { opr = "add", problemStartTime = 3 } : i32
  "dummy.store_A"(%2) { opr = "mem_port", problemStartTime = 4 } : (i32) -> ()
  return { problemStartTime = 5 }
}
```

Here is the same test-case encoded in the SSP dialect:

```mlir
ssp.instance "canis14_fig2" of "ModuloProblem" [II<3>] {
  operator_type @MemPort [latency<1>, limit<1>]
  operator_type @Add [latency<1>]
  operator_type @Implicit [latency<0>]
  %0 = operation<@MemPort>(@store_A [dist<1>]) [t<2>]
  %1 = operation<@MemPort>() [t<0>]
  %2 = operation<@Add>(%0, %1) [t<3>]
  operation<@MemPort> @store_A(%2) [t<4>]
  operation<@Implicit>(@store_A) [t<5>]
}
```

Emitting an SSP dump is also useful to test that an conversion pass correctly constructs the scheduling problem, e.g. checking that it contains a memory dependence to another operation:

```mlir
// CHECK: operation<@{{.*}}>(%0, @[[store_1]])
%5 = affine.load %0[0] : memref<1xi32>
...
// CHECK: operation<@{{.*}}> @[[store_1:.*]](%7, %0)
affine.store %7, %0[0] : memref<1xi32>
```

### Benchmarking

### Rapid prototyping





### Q&A
- Do I need it?
- Why not Capnproto
## Naming

## Representation of problem components

### Instance

### Operator type

### Operation

### Dependence

## Import/export

## Extensibility
