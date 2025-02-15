//===-- Passes.td - Transforms pass definition file --------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains definitions for passes within the Transforms/ directory.
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_TRANSFORMS_PASSES
#define CIRCT_TRANSFORMS_PASSES

include "mlir/Pass/PassBase.td"
include "mlir/Rewrite/PassUtil.td"

def FlattenMemRef : Pass<"flatten-memref", "::mlir::ModuleOp"> {
  let summary = "Flatten memrefs";
  let description = [{
    Flattens multidimensional memories and accesses to them into
    single-dimensional memories.}];
  let constructor = "circt::createFlattenMemRefPass()";
  let dependentDialects = ["mlir::memref::MemRefDialect"];
}

def FlattenMemRefCalls : Pass<"flatten-memref-calls", "::mlir::ModuleOp"> {
  let summary = "Flatten memref calls";
  let description = [{
    Flattens calls to functions which have multidimensional memrefs as arguments.
    This is done by casting memref arguments through memref.subview operations.
    Any called functions which had their type signatures changes will be replaced
    by a private function definition with the new signature.
    It is up to users of this pass to define how these rewritten functions are
    to be implemented.}];
  let constructor = "circt::createFlattenMemRefCallsPass()";
  let dependentDialects = ["mlir::memref::MemRefDialect"];
}

def StripDebugInfoWithPred : Pass<"strip-debuginfo-with-pred", "::mlir::ModuleOp"> {
  let summary = "Selectively strip debug info from all operations";

  let description = [{
    This pass extends mlir::StripDebugInfoPass to selectively strip locations with a
    given predicate.
  }];
  let constructor = "circt::createStripDebugInfoWithPredPass({})";
  let options = [
    Option<"dropSuffix", "drop-suffix", "std::string",
           /*default=*/"",
           "Drop file location info with the specified suffix. This option is"
           "intended to be used for testing."> ];
}

def MapArithToCombPass : Pass<"map-arith-to-comb"> {
  let summary = "Map arith ops to combinational logic";
  let description = [{
    A pass which does a simple `arith` to `comb` mapping wherever possible.
    This pass will not convert:  
    * floating point operations
    * operations using `vector`-typed values

    This **does not** intend to be the definitive lowering/HLS pass of `arith`
    operations in CIRCT (hence the name "map" instead of e.g. "lower").
    Rather, it provides a simple way (mostly for testing purposes) to map
    `arith` operations.
  }];
  let constructor = "circt::createMapArithToCombPass()";
  let dependentDialects = ["circt::hw::HWDialect, mlir::arith::ArithDialect, circt::comb::CombDialect"];
}

def InsertMergeBlocks : Pass<"insert-merge-blocks", "::mlir::ModuleOp"> {
  let summary = "Insert explicit merge blocks";
  let description = [{
    This pass inserts additional merge blocks for each block with more than
    two successors. A merge block is a block that only contains one operation,
    a terminator, and has two predecessors.
    The order successors are merged together mirrors the order different control
    paths are created. Thus, each block with two successors will have a corresponding
    merge block.

    This pass brings the CFG into a canonical form for further transformations.

    Treats loops and sub-CFGs with irregular control flow like single blocks.
  }];
  let constructor = "circt::createInsertMergeBlocksPass()";
  let dependentDialects = ["mlir::cf::ControlFlowDialect", "mlir::func::FuncDialect"];
}

def MaximizeSSA : Pass<"maximize-ssa", "::mlir::ModuleOp"> {
  let summary = "Convert every function in the module into maximal SSA form";
  let description = [{
    Convert the region within every function into maximal SSA form. This
    ensures that every value used within a block is also defined within the
    block, making dataflow explicit and removing block dominance-based dataflow
    semantics. The pass achieves this by adding block arguments wherever
    necessary to forward values to the block(s) where they are used.
  }];
  let constructor = "circt::createMaximizeSSAPass()";
}

def PrintOpCount : Pass<"print-op-count", "::mlir::ModuleOp"> {
  let summary = "Print operation count analysis results";
  let description = [{
    This pass prints data on operation counts in a builtin.module.
  }];
  let constructor = "circt::createPrintOpCountPass()";
  let options = [
    Option<"emissionFormat", "emission-format",
           "circt::OpCountEmissionFormat",
           "circt::OpCountEmissionFormat::Readable",
           "Specify the format to emit op count info in",
           [{::llvm::cl::values(
            clEnumValN(circt::OpCountEmissionFormat::Readable,
              "readable", "Emit the data in human-readable form"),
            clEnumValN(circt::OpCountEmissionFormat::ReadableSorted,
              "readable-sorted", "Emit the data in human-readable form, sorted for output determinism"),
            clEnumValN(circt::OpCountEmissionFormat::JSON,
              "json", "Emit the data in JSON form")
          )}]
           >
  ];
}

def MemoryBanking : Pass<"memory-banking", "::mlir::func::FuncOp"> {
  let summary = "Partition the memories used in affine parallel loops into banks";
  let constructor = "circt::createMemoryBankingPass()";
  let options = [
    Option<"bankingFactor", "factor", "unsigned", /*default=*/"1",
           "Use this banking factor for all memories being partitioned">,
    Option<"bankingDimension", "dimension", "int", /*default=*/"-1",
           "The dimension along which to bank the memory. If -1, the innermost dimension with size > 1 is used.">
  ];
  let dependentDialects = ["mlir::memref::MemRefDialect, mlir::scf::SCFDialect, mlir::affine::AffineDialect"];
}

def IndexSwitchToIf : Pass<"switch-to-if", "::mlir::ModuleOp"> {
  let summary = "Index switch to if";
  let description = [{
      Transform `scf.index_switch` to a series of `scf.if` operations.
      This is necessary for dialects that don't support switch statements, e.g., Calyx.
      An example:
      ```
        %0 = scf.index_switch %cond -> i32
          case 0 { ... }
          case 1 { ... }
          ...

        =>

        %c0 = arith.cmpi eq %0, 0
        %c1 = arith.cmpi eq %0, 1
        %0 = scf.if %c0 {
         ...
        } else {
          %1 = scf.if %c1 {
            ...
          } else {
            ...
          }
        }
      ```
  }];
  let constructor = "circt::createIndexSwitchToIfPass()";
  let dependentDialects = ["mlir::scf::SCFDialect"];
}

def HierarchicalRunner : Pass<"hierarchical-runner", "::mlir::ModuleOp"> {
  let summary = "Run passes under hierarchy";
  let description = [{
    Run passes under hierarchy.
  }];
  let options = [
    Option<"pipelineStr", "pipeline", "std::string", "",
           "The pipeline to run under hierarchy">,
    Option<"topName", "top-name", "std::string", "",
           "The name of the top-level module to run the pass on">,
    Option<"includeBoundInstances", "include-bound-instances", "bool", "false",
           "Whether to include bound instances in the hierarchy">
  ];
}

#endif // CIRCT_TRANSFORMS_PASSES
