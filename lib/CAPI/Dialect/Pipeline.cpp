//===- Pipeline.cpp - C interface for the Pipeline dialect ----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "circt-c/Dialect/Pipeline.h"
#include "circt/Dialect/Pipeline/PipelineDialect.h"
#include "circt/Dialect/Pipeline/PipelinePasses.h"

#include "mlir/CAPI/Registration.h"

using namespace circt::pipeline;

MLIR_DEFINE_CAPI_DIALECT_REGISTRATION(Pipeline, pipeline,
                                      circt::pipeline::PipelineDialect)

void registerPipelinePasses() { circt::pipeline::registerPasses(); }
