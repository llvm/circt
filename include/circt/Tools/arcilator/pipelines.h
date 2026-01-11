//===- pipelines.h - Arcilator lowering pipelines -------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file declares the 'arcilator' lowering pipelines.
//
//===----------------------------------------------------------------------===//

#ifndef TOOLS_ARCILATOR_PIPELINES_H
#define TOOLS_ARCILATOR_PIPELINES_H

#include "mlir/Pass/PassManager.h"
#include "mlir/Pass/PassOptions.h"
#include "llvm/Support/CommandLine.h"

namespace circt {

// Pre-process the input such that it no longer contains any SV dialect ops
// and external modules that are relevant to the arc transformation are
// represented as intrinsic ops.
struct ArcPreprocessingOptions
    : mlir::PassPipelineOptions<ArcPreprocessingOptions> {
  Option<bool> observePorts{*this, "observe-ports",
                            llvm::cl::desc("Make all ports observable"),
                            llvm::cl::init(false)};

  Option<bool> observeWires{*this, "observe-wires",
                            llvm::cl::desc("Make all wires observable"),
                            llvm::cl::init(false)};

  Option<bool> observeNamedValues{
      *this, "observe-named-values",
      llvm::cl::desc("Make values with `sv.namehint` observable"),
      llvm::cl::init(false)};

  Option<bool> observeMemories{
      *this, "observe-memories",
      llvm::cl::desc("Make all memory contents observable"),
      llvm::cl::init(false)};

  Option<bool> asyncResetsAsSync{
      *this, "async-resets-as-sync",
      llvm::cl::desc("Treat asynchronous firreg resets as synchronous"),
      llvm::cl::init(false)};
};
void populateArcPreprocessingPipeline(
    mlir::OpPassManager &pm, const ArcPreprocessingOptions &options = {});

// Restructure the input from a `hw.module` hierarchy to a collection of arcs.
struct ArcConversionOptions : mlir::PassPipelineOptions<ArcConversionOptions> {
  Option<bool> observeRegisters{*this, "observe-registers",
                                llvm::cl::desc("Make all registers observable"),
                                llvm::cl::init(false)};

  Option<bool> shouldDedup{*this, "dedup", llvm::cl::desc("Deduplicate arcs"),
                           llvm::cl::init(true)};
};
void populateArcConversionPipeline(mlir::OpPassManager &pm,
                                   const ArcConversionOptions &options = {});

// Perform arc-level optimizations that are not specific to software
// simulation.
struct ArcOptimizationOptions
    : mlir::PassPipelineOptions<ArcOptimizationOptions> {
  Option<bool> shouldDedup{*this, "dedup", llvm::cl::desc("Deduplicate arcs"),
                           llvm::cl::init(true)};

  Option<bool> shouldDetectEnables{
      *this, "detect-enables",
      llvm::cl::desc("Infer enable conditions for states to avoid computation"),
      llvm::cl::init(true)};

  Option<bool> shouldDetectResets{
      *this, "detect-resets",
      llvm::cl::desc("Infer reset conditions for states to avoid computation"),
      llvm::cl::init(false)};

  Option<bool> shouldMakeLUTs{
      *this, "lookup-tables",
      llvm::cl::desc("Optimize arcs into lookup tables"), llvm::cl::init(true)};
};
void populateArcOptimizationPipeline(
    mlir::OpPassManager &pm, const ArcOptimizationOptions &options = {});

// Lower stateful arcs into explicit state reads and writes.
struct ArcStateLoweringOptions
    : mlir::PassPipelineOptions<ArcStateLoweringOptions> {
  Option<bool> shouldInline{*this, "inline", llvm::cl::desc("Inline arcs"),
                            llvm::cl::init(true)};
};
void populateArcStateLoweringPipeline(
    mlir::OpPassManager &pm, const ArcStateLoweringOptions &options = {});

// Allocate states.
struct ArcStateAllocationOptions
    : mlir::PassPipelineOptions<ArcStateAllocationOptions> {
  Option<unsigned> splitFuncsThreshold{
      *this, "split-funcs-threshold",
      llvm::cl::desc("Split large MLIR functions that occur above the given "
                     "size threshold"),
      llvm::cl::ValueOptional};
};
void populateArcStateAllocationPipeline(
    mlir::OpPassManager &pm, const ArcStateAllocationOptions &options = {});

// Lower the arcs and update functions to LLVM. This pipeline lowers modules to
// LLVM IR.
void populateArcToLLVMPipeline(mlir::OpPassManager &pm, bool insertRuntime,
                               mlir::StringRef extraRuntimeArgs = {});

} // namespace circt

#endif // TOOLS_ARCILATOR_PIPELINE_H
