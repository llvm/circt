# HLS in CIRCT

// write a compelling introduction here

[TOC]

## `hlstool`

[`hlstool`](https://github.com/llvm/circt/blob/main/tools/hlstool/hlstool.cpp) is a tool for driving various CIRCT-based HLS flows. The tool works by defining [pass pipelines](https://mlir.llvm.org/docs/PassManagement/) that string together various MLIR and CIRCT passes to realize an HLS flow.  
For new users to MLIR, it is important to recognize the different between such compiler driver tools, and the MLIR `opt` tools (optimization drivers), such as `mlir-opt` and `circt-opt`.  
For instance, in `hlstool` we may specify a pass pipeline such as:
```c++
  // Memref legalization.
  pm.addPass(circt::createFlattenMemRefPass());
  pm.nest<func::FuncOp>().addPass(
      circt::handshake::createHandshakeLegalizeMemrefsPass());
  pm.addPass(mlir::createConvertSCFToCFPass());
  pm.nest<handshake::FuncOp>().addPass(createSimpleCanonicalizerPass());
```
For the compiler developer, however, we often *don't* want to run the entire HLS pass pipeline on our IR, but instead work on a specific level of IR and run a specific pass. Thus, every pass that is nested within `hlstool` can also be driven by an `opt` tool, e.g. `circt::createFlattenMemRefPass` is registered in `circt-opt` as `circt-opt --flatten-memref`.

## Flows

CIRCT homes multiple actively developed HLS flows, each with their own approach to HLS. This section aims to provide a brief overview of the different flows, and how to use them.  

**Note:** we do mean 'active development'**(!)** - most of these flows are the combined efforts of various student, phd and research projects, and should **not** be considered production-ready. You **will** encounter bugs, but you will also have a group of people are happy to help you fix them, and improve the flows. Feel free to reach out to the [codeowner](https://github.com/llvm/circt/blob/main/CODEOWNERS) of the dialect/pass that you're working on, or ask a question in the [LLVM discord](https://discord.gg/Jsktb5PR) which has a dedicated `circt` channel.

How flow documentation is structured:
* **Description**: A brief description/rationale of the flow, what it is and why it is how it is.
* **Relevant dialects**: A listing of the dialects used in the flow (preferably in order), as well as links to the dialect rationales. We encourage the user to read the dialect rationales to understand the design decisions behind the dialects.
* **Example usage/relevant tests**:  A vast majority of CIRCTs flow-specific knowledge is (unfortunately) institutional. However, users are often referred to the CIRCT tests to understand what different passes do, and how to drive the various CIRCT tools to achieve their goals. This section thus lists some relevant tests that may serve as important references of how to use the flow.

## Dynamically scheduled HLS (DHLS)



#### Relevant dialects:
- [cf](https://mlir.llvm.org/docs/Dialects/ControlFlowDialect/)
- [Handshake](https://circt.llvm.org/docs/Dialects/Handshake/RationaleHandshake/)

#### Example usage/Relevant tests
- [handshake integration tests](https://github.com/llvm/circt/tree/main/integration_test/Dialect/Handshake) is a collection of cocotb-based tests that use `hlstool` to convert `cf`-level code to verilog.
- [handshake-runner](https://github.com/llvm/circt/tree/main/integration_test/handshake-runner) tests show how to use the `handshake-runner` to interpret `handshake`-level IR

#### Resources

The following is a list of work that has used or contributed to the DHLS flow in CIRCT:
- (10/2022) [Multi-Level Rewriting for Stream Processing to RTL compilation (M.Sc. Thesis) - Christian Ulmann](https://www.research-collection.ethz.ch/handle/20.500.11850/578713)
- (01/2022) [A Dynamically Scheduled HLS Flow in MLIR (M.Sc. Thesis) - Morten Borup Petersen](https://infoscience.epfl.ch/record/292189)


## Calyx

#### Relevant dialects:

#### Example usage/Relevant tests

#### Resources
