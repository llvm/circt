//===- RTGPipeline.cpp - C Interface to random test generation ------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "circt-c/RTGPipeline.h"

#include "circt/Dialect/RTG/Transforms/RTGPasses.h"
#include "circt/Support/Passes.h"
#include "circt/Target/EmitRTGAssembly.h"
#include "mlir/CAPI/IR.h"
#include "mlir/CAPI/Support.h"
#include "mlir/CAPI/Utils.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Tools/Plugins/DialectPlugin.h"
#include "llvm/Support/raw_ostream.h"

using namespace mlir;
using namespace circt;

MlirLogicalResult circtGenerateRandomTests(
    MlirModule module, bool verifyPasses, bool verbosePassExecution,
    bool hasSeed, unsigned seed, unsigned numUnsupportedInstructions,
    char **unsupportedInstructions, const char *unsupportedInstructionsFile,
    CirctRTGOutputFormat outputFormat, MlirStringCallback callback,
    void *userData) {
  mlir::detail::CallbackOstream stream(callback, userData);

  auto moduleOp = unwrap(module);

  PassManager pm(moduleOp.getContext());
  pm.enableVerifier(verifyPasses);

  if (verbosePassExecution)
    pm.addInstrumentation(
        std::make_unique<VerbosePassInstrumentation<mlir::ModuleOp>>(
            "rtgtool"));

  pm.addPass(createSimpleCanonicalizerPass());
  if (outputFormat != CirctRTGOutputMLIR) {
    rtg::ElaborationOptions options;
    if (hasSeed)
      options.seed = seed;
    pm.addPass(rtg::createElaborationPass(options));
    pm.addPass(createSimpleCanonicalizerPass());
  }

  if (failed(pm.run(moduleOp)))
    return wrap(failure());

  if (outputFormat == CirctRTGOutputMLIR ||
      outputFormat == CirctRTGOutputRenderedMLIR) {
    OpPrintingFlags printingFlags;
    moduleOp->print(stream, printingFlags);
    return wrap(success());
  }

  EmitRTGAssembly::EmitRTGAssemblyOptions options;
  SmallVector<std::string> instr;
  for (unsigned i = 0; i < numUnsupportedInstructions; ++i)
    instr.push_back(std::string(unsupportedInstructions[i]));
  options.unsupportedInstructions = instr;
  std::string file(unsupportedInstructionsFile);
  EmitRTGAssembly::parseUnsupportedInstructionsFile(
      file, options.unsupportedInstructions);
  return wrap(EmitRTGAssembly::emitRTGAssembly(moduleOp, stream, options));
}
