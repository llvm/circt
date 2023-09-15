//===- PipelineOps.h - Pipeline dialect operations ---------------- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_DIALECT_PIPELINE_PIPELINEOPS_H
#define CIRCT_DIALECT_PIPELINE_PIPELINEOPS_H

#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/RegionKindInterface.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Interfaces/CallInterfaces.h"
#include "mlir/Interfaces/ControlFlowInterfaces.h"
#include "mlir/Interfaces/InferTypeOpInterface.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

#include "circt/Dialect/Pipeline/PipelineDialect.h"
#include "circt/Dialect/Seq/SeqTypes.h"

namespace circt {
namespace pipeline {
class StageOp;
class ScheduledPipelineOp;

// StageKind defines the control semantics of a pipeline stages.
enum class StageKind {
  // All stages in a pipeline without a stall signal is a continuous stage.
  Continuous,
  // Stallable stages are any stages which appear **before** the first
  // non-stallable stage in the pipeline.
  Stallable,
  // Non-stallable stages are the stages explicitly marked as non-stallable by
  // the user.
  NonStallable,
  // Runoff stages and stages that appear **after** (and by extension,
  // **between** non-stallable stages). Runoff stages consider their own
  // enablement wrt. the stall signal, as well as the enablement of the **last
  // non-stallable register** (LNS) wrt. the runoff stage's position in the
  // pipeline.
  Runoff
};

namespace detail {

// Returns the set of values defined outside of the given region, and the
// operation that defines the region. This will walk the entire region so
// should be used with care (or cache the results).
llvm::SmallVector<Value> getValuesDefinedOutsideRegion(Region &region);
} // namespace detail

// Determines the stage which 'op' resides in within the pipeline. This is
// useful for analysis of the pipeline, wherein ops may reside in nested
// regions within different stages of the pipeline.
Block *getParentStageInPipeline(ScheduledPipelineOp pipeline, Operation *op);

// Determines the stage which 'block' resides in within the pipeline. This is
// useful for analysis of the pipeline, wherein blocks may reside in nested
// regions within different stages of the pipeline.
Block *getParentStageInPipeline(ScheduledPipelineOp pipeline, Block *block);

// Determines the stage which 'v' resides in within the pipeline. This is
// useful for analysis of the pipeline, wherein values may reside in nested
// regions within different stages of the pipeline.
Block *getParentStageInPipeline(ScheduledPipelineOp pipeline, Value v);

} // namespace pipeline
} // namespace circt

#define GET_OP_CLASSES
#include "circt/Dialect/Pipeline/Pipeline.h.inc"

#endif // CIRCT_DIALECT_PIPELINE_PIPELINEOPS_H
