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

#include "llvm/ADT/SmallSet.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/StringMap.h"

using namespace circt;
using namespace msft;
using GeneratorSet = llvm::SmallSet<StringRef, 8>;

namespace {
/// Holds the set of registered generators for each operation.
class OpGenerator {
  llvm::StringMap<GeneratorCallback> generators;

public:
  void registerOpGenerator(StringRef generatorName, GeneratorCallback cb) {
    generators[generatorName] = cb;
  }

  LogicalResult runOnOperation(mlir::Operation *op, GeneratorSet generatorSet);
};

} // namespace

LogicalResult OpGenerator::runOnOperation(mlir::Operation *op,
                                          GeneratorSet generatorSet) {
  if (generators.size() == 0)
    return failure();

  // Select the generator specified in the generator set. If none were
  // specified, take the first generator. If multiple candidate generators are
  // present, raise an error.
  GeneratorCallback gen;
  for (auto &generatorPair : generators) {
    if (generatorSet.contains(generatorPair.first())) {
      if (gen)
        return op->emitError("multiple generators selected");
      gen = generatorPair.second;
    }
  }
  if (!gen)
    gen = generators.begin()->second;

  mlir::IRRewriter rewriter(op->getContext());
  Operation *replacement = gen(op);
  if (replacement == nullptr)
    return op->emitError("Failed generator on ") << op->getName();
  rewriter.replaceOp(op, replacement->getResults());
  return success();
}

namespace circt {
namespace msft {
namespace detail {
struct Generators {
  llvm::StringMap<OpGenerator> registeredOpGenerators;

  LogicalResult runOnOperation(mlir::Operation *op, GeneratorSet generatorSet) {
    StringRef opName = op->getName().getStringRef();
    auto opGenerator = registeredOpGenerators.find(opName);
    if (opGenerator == registeredOpGenerators.end())
      return success(); // If we don't have a generator registered, just return.
    return opGenerator->second.runOnOperation(op, generatorSet);
  }
};
} // namespace detail
} // namespace msft
} // namespace circt

void MSFTDialect::registerGenerator(StringRef opName, StringRef generatorName,
                                    GeneratorCallback cb) {
  generators->registeredOpGenerators[opName].registerOpGenerator(generatorName,
                                                                 cb);
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
  GeneratorSet generatorSet;
  for (auto &generator : generators)
    generatorSet.insert(StringRef(generator));
  MLIRContext *ctxt = &getContext();
  MSFTDialect *msft = ctxt->getLoadedDialect<MSFTDialect>();
  if (!msft)
    return;

  Operation *top = getOperation();
  top->walk([this, msft, generatorSet](Operation *op) {
    if (failed(msft->generators->runOnOperation(op, generatorSet)))
      signalPassFailure();
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
