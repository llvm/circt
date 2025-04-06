//===- RtgTool.cpp - C Interface for the rtgtool --------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "circt-c/RtgTool.h"

#include "circt/Tools/rtgtool/RtgToolOptions.h"
#include "mlir/CAPI/Pass.h"
#include <string>

using namespace circt;
using namespace circt::rtg;

//===----------------------------------------------------------------------===//
// Tool Option API.
//===----------------------------------------------------------------------===//

DEFINE_C_API_PTR_METHODS(CirctRtgToolOptions, RtgToolOptions)

CirctRtgToolOptions circtRtgToolOptionsCreateDefault(unsigned seed) {
  auto *options = new RtgToolOptions(seed);
  return wrap(options);
}

void circtRtgToolOptionsDestroy(CirctRtgToolOptions options) {
  delete unwrap(options);
}

void circtRtgToolOptionsSetOutputFormat(CirctRtgToolOptions options,
                                        CirctRtgToolOutputFormat format) {
  RtgToolOptions::OutputFormat converted;
  switch (format) {
  case CIRCT_RTGTOOL_OUTPUT_FORMAT_MLIR:
    converted = RtgToolOptions::OutputFormat::MLIR;
    break;
  case CIRCT_RTGTOOL_OUTPUT_FORMAT_ELABORATED_MLIR:
    converted = RtgToolOptions::OutputFormat::ElaboratedMLIR;
    break;
  case CIRCT_RTGTOOL_OUTPUT_FORMAT_ASM:
    converted = RtgToolOptions::OutputFormat::ASM;
    break;
  }

  unwrap(options)->setOutputFormat(converted);
}

void circtRtgToolOptionsSetSeed(CirctRtgToolOptions options, unsigned seed) {
  unwrap(options)->setSeed(seed);
}

void circtRtgToolOptionsSetVerifyPasses(CirctRtgToolOptions options,
                                        bool enable) {
  unwrap(options)->setVerifyPasses(enable);
}

void circtRtgToolOptionsSetVerbosePassExecution(CirctRtgToolOptions options,
                                                bool enable) {
  unwrap(options)->setVerbosePassExecution(enable);
}

void circtRtgToolOptionsSetUnsupportedInstructions(
    CirctRtgToolOptions options, unsigned numInstr,
    const void **unsupportedInstructions) {
  SmallVector<std::string> instr;
  for (unsigned i = 0; i < numInstr; ++i)
    instr.push_back(
        reinterpret_cast<std::string *>(unsupportedInstructions)[i]);
  unwrap(options)->setUnsupportedInstructions(std::move(instr));
}

void circtRtgToolOptionsAddUnsupportedInstruction(
    CirctRtgToolOptions options, const char *unsupportedInstruction) {
  unwrap(options)->addUnsupportedInstruction(
      std::string(unsupportedInstruction));
}

void circtRtgToolOptionsSetUnsupportedInstructionsFile(
    CirctRtgToolOptions options, const char *filename) {
  unwrap(options)->setUnsupportedInstructionsFile(std::string(filename));
}

void circtRtgToolOptionsSetSplitOutput(CirctRtgToolOptions options,
                                       bool enable) {
  unwrap(options)->setSplitOutput(enable);
}

void circtRtgToolOptionsSetOutputPath(CirctRtgToolOptions options,
                                      const char *path) {
  unwrap(options)->setOutputPath(std::string(path));
}

void circtRtgToolOptionsSetMemoriesAsImmediates(CirctRtgToolOptions options,
                                                bool enable) {
  unwrap(options)->setMemoriesAsImmediates(enable);
}

//===----------------------------------------------------------------------===//
// Pipeline Population API.
//===----------------------------------------------------------------------===//

void circtRtgToolRandomizerPipeline(MlirPassManager pm,
                                    CirctRtgToolOptions options) {
  populateRandomizerPipeline(*unwrap(pm), *unwrap(options));
}
