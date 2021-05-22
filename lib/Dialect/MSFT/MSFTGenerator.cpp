//===- MSFTGenerator.cpp - Implement MSFT generators ----------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/MSFT/MSFTDialect.h"

#include "mlir/IR/ImplicitLocOpBuilder.h"
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
  ImplicitLocOpBuilder builder = ImplicitLocOpBuilder::atBlockEnd(
      op->getLoc(), op->getParentOfType<ModuleOp>().getBody());

  StringAttr name = builder.getStringAttr(op->getName().getStringRef());
  SmallVector<hw::ModulePortInfo, 8> ports;
  // for (auto operand : op->getOperands())
  //   ports.push_back(hw::ModulePortInfo{});
  hw::HWModuleOp intoMod = builder.create<hw::HWModuleOp>(name, ports);

  auto result = generate(op, intoMod);
  if (failed(result))
    op->emitError("Failed generator on ") << op->getName();
  return result;
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

static constexpr StringRef DesignEntryPrefix = "circt.design_entry.";
void RunGeneratorsPass::runOnOperation() {
  Operation *top = getOperation();
  top->walk([this](Operation *op) {
    StringRef opName = op->getName().getStringRef();
    if (!opName.startswith(DesignEntryPrefix))
      return;
    StringRef designModuleName = opName.substr(DesignEntryPrefix.size());
    auto opGenerator = registeredOpGenerators.find(designModuleName);
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
