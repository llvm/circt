//===- Pipeline.h - Pipeline Definitions ------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines an Pipeline MLIR dialect.
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_PIPELINE_OPS_H_
#define CIRCT_PIPELINE_OPS_H_

#include "circt/Dialect/ESI/ESITypes.h"
#include "circt/Support/LLVM.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/RegionKindInterface.h"
#include "mlir/IR/TypeSupport.h"
#include "mlir/IR/Types.h"
#include "mlir/Interfaces/ControlFlowInterfaces.h"
#include "mlir/Interfaces/InferTypeOpInterface.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Pass/Pass.h"

#include "circt/Dialect/Pipeline/PipelineDialect.h.inc"

#define GET_OP_CLASSES

namespace circt {
namespace pipeline {
class StageOp;
class ScheduledPipelineOp;

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

#include "circt/Dialect/Pipeline/PipelineInterfaces.h.inc"

#include "circt/Dialect/Pipeline/Pipeline.h.inc"

#endif // CIRCT_PIPELINE_OPS_H_
