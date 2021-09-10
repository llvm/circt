//===- MSFTPasses.cpp - Implement MSFT passes -----------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/MSFT/MSFTDialect.h"
#include "circt/Dialect/MSFT/MSFTOps.h"

#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/Transforms/DialectConversion.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallSet.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/StringMap.h"

using namespace circt;
using namespace msft;
using GeneratorSet = llvm::SmallSet<StringRef, 8>;

//===----------------------------------------------------------------------===//
// Generators.
//===----------------------------------------------------------------------===//

namespace {
/// Holds the set of registered generators for each operation.
class OpGenerator {
  llvm::StringMap<llvm::DenseMap<Attribute, GeneratorCallback>> generators;

public:
  void registerOpGenerator(StringRef generatorName, Attribute parameters,
                           GeneratorCallback cb) {
    generators[generatorName][parameters] = cb;
  }

  LogicalResult runOnOperation(mlir::Operation *op, GeneratorSet generatorSet);
};

} // namespace

LogicalResult OpGenerator::runOnOperation(mlir::Operation *op,
                                          GeneratorSet generatorSet) {
  if (generators.size() == 0)
    return failure();
  if (generators.size() > 1 && generatorSet.size() < 1)
    return op->emitError(op->getName().getStringRef())
           << " at least one generator must be selected";

  // Check if any of the generators were selected in the generator set. If more
  // than one candidate is present in the generator set, raise an error.
  GeneratorCallback gen;
  Attribute parameters = op->getAttr("parameters");
  for (auto &generatorPair : generators) {
    if (generatorSet.contains(generatorPair.first())) {
      if (gen)
        return op->emitError("multiple generators selected");
      auto callbackPair = generatorPair.second.find(parameters);
      if (callbackPair != generatorPair.second.end())
        gen = callbackPair->second;
    }
  }

  // If no generator was selected by the generator set, and there is just one
  // generator, default to using that. Otherwise raise an error.
  if (!gen) {
    if (generators.size() == 1) {
      auto generatorMap = generators.begin()->second;
      auto callbackPair = generatorMap.find(parameters);
      if (callbackPair != generatorMap.end())
        gen = callbackPair->second;
    } else {
      return op->emitError("unable to select a generator");
    }
  }

  mlir::IRRewriter rewriter(op->getContext());
  Operation *replacement = gen(op);
  if (replacement == nullptr)
    return op->emitError("Failed generator on ") << op->getName();
  replacement->setLoc(op->getLoc());
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
                                    GeneratorCallback cb,
                                    Attribute parameters) {
  generators->registeredOpGenerators[opName].registerOpGenerator(
      generatorName, parameters, cb);
}

namespace circt {
namespace msft {
#define GEN_PASS_CLASSES
#include "circt/Dialect/MSFT/MSFTPasses.h.inc"
} // namespace msft
} // namespace circt

namespace {
struct RunGeneratorsPass : public RunGeneratorsBase<RunGeneratorsPass> {
  void runOnOperation() override;
};
} // anonymous namespace

void RunGeneratorsPass::runOnOperation() {
  GeneratorSet generatorSet;
  for (auto &generator : generators)
    if (!generator.empty())
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

//===----------------------------------------------------------------------===//
// Lower MSFT to HW.
//===----------------------------------------------------------------------===//

namespace {
/// Lower `ChannelBuffer`s, breaking out the various options. For now, just
/// replace with the specified number of pipeline stages (since that's the only
/// option).
struct InstanceOpLowering : public OpConversionPattern<InstanceOp> {
public:
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(InstanceOp, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const final;
};
} // anonymous namespace

LogicalResult
InstanceOpLowering::matchAndRewrite(InstanceOp msftInst,
                                    ArrayRef<Value> operands,
                                    ConversionPatternRewriter &rewriter) const {
  auto hwInst = rewriter.create<hw::InstanceOp>(
      msftInst.getLoc(), msftInst.getResultTypes(), msftInst.instanceNameAttr(),
      msftInst.moduleNameAttr(), operands, DictionaryAttr{}, StringAttr{});
  rewriter.replaceOp(msftInst, hwInst.getResults());
  return success();
}

namespace {
struct LowerToHWPass : public LowerToHWBase<LowerToHWPass> {
  void runOnOperation() override;
};
} // anonymous namespace

void LowerToHWPass::runOnOperation() {
  auto top = getOperation();
  auto ctxt = &getContext();

  // Set up a conversion and give it a set of laws.
  ConversionTarget target(*ctxt);
  target.addIllegalDialect<MSFTDialect>();
  target.addLegalDialect<hw::HWDialect>();

  // Add all the conversion patterns.
  RewritePatternSet patterns(ctxt);
  patterns.insert<InstanceOpLowering>(ctxt);

  // Run the conversion.
  if (failed(applyPartialConversion(top, target, std::move(patterns))))
    signalPassFailure();
}

namespace circt {
namespace msft {
std::unique_ptr<Pass> createLowerToHWPass() {
  return std::make_unique<LowerToHWPass>();
}
} // namespace msft
} // namespace circt

namespace {
#define GEN_PASS_REGISTRATION
#include "circt/Dialect/MSFT/MSFTPasses.h.inc"
} // namespace

void circt::msft::registerMSFTPasses() { registerPasses(); }
