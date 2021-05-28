//===- MSFTGenerator.cpp - Implement MSFT generators ----------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/MSFT/MSFTDialect.h"

#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassRegistry.h"

#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/StringMap.h"

using namespace circt;
using namespace msft;

namespace {

class GeneratorName {
  GeneratorCallback generate;

public:
  GeneratorName() : generate(nullptr) {}
  GeneratorName(GeneratorCallback cb) : generate(cb) {}

  LogicalResult runOnOperation(mlir::Operation *op);
};

LogicalResult GeneratorName::runOnOperation(mlir::Operation *op) {
  mlir::IRRewriter rewriter(op->getContext());
  Operation *replacement = generate(op);
  if (replacement == nullptr)
    return op->emitError("Failed generator on ") << op->getName();
  rewriter.replaceOp(op, replacement->getResults());
  return success();
}

class OpGenerator {
  llvm::StringMap<GeneratorName> generators;

public:
  void registerOpGenerator(StringRef generatorName, GeneratorCallback cb) {
    generators[generatorName] = GeneratorName(cb);
  }

  LogicalResult runOnOperation(mlir::Operation *op) {
    // Use the first one, if there is one registered.
    if (generators.size() > 0)
      return generators.begin()->second.runOnOperation(op);
    return failure();
  }
};

} // namespace

static llvm::StringMap<OpGenerator> registeredOpGenerators;

void circt::msft::registerGenerator(StringRef opName, StringRef generatorName,
                                    GeneratorCallback cb) {
  registeredOpGenerators[opName].registerOpGenerator(generatorName, cb);
}

namespace circt {
namespace msft {
#define GEN_PASS_CLASSES
#include "circt/Dialect/MSFT/MSFTPasses.h.inc"
} // namespace msft
} // namespace circt

namespace {
/// Run all the physical lowerings.
struct RunGeneratorsPass : public RunGeneratorsBase<RunGeneratorsPass> {
  void runOnOperation() override;
};
} // anonymous namespace

void RunGeneratorsPass::runOnOperation() {
  Operation *top = getOperation();
  top->walk([this](Operation *op) {
    StringRef opName = op->getName().getStringRef();
    auto opGenerator = registeredOpGenerators.find(opName);
    if (opGenerator != registeredOpGenerators.end()) {
      auto rc = opGenerator->second.runOnOperation(op);
      if (failed(rc))
        signalPassFailure();
    }
  });
}

namespace circt {
namespace msft {
std::unique_ptr<Pass> createRunGeneratorsPass() {
  return std::make_unique<RunGeneratorsPass>();
}

} // namespace msft
} // namespace circt

namespace {
#define GEN_PASS_REGISTRATION
#include "circt/Dialect/MSFT/MSFTPasses.h.inc"
} // namespace

void circt::msft::registerMSFTPasses() { registerPasses(); }
