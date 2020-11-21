//===- ESIToRTL.cpp - ESI to RTL/SV conversion passes -----------*- C++ -*-===//
//
// Lower ESI to RTL and SV.
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/ESI/ESIDialect.h"
#include "circt/Dialect/ESI/ESIOps.h"
#include "circt/Dialect/RTL/Dialect.h"
#include "circt/Dialect/SV/Dialect.h"

#include "mlir/IR/Builders.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Support/Debug.h"

#include <memory>

#define DEBUG_TYPE "esi-to-rtl"

namespace circt {
namespace esi {
#define GEN_PASS_CLASSES
#include "circt/Dialect/ESI/ESIPasses.h.inc"
} // namespace esi
} // namespace circt

using namespace mlir;
using namespace circt::esi;

namespace {
struct ESIToPhysicalPass : public LowerESIToPhysicalBase<ESIToPhysicalPass> {

  void runOnOperation() override;

private:
  void expandBuffer(ChannelBuffer buffer);

  OpBuilder *builder = nullptr;
};

void ESIToPhysicalPass::runOnOperation() {
  OpBuilder theBuilder(&getContext());
  builder = &theBuilder;

  auto *moduleBody = getOperation().getBody();
  SmallVector<ChannelBuffer, 4> toExpand;
  for (auto &op : *moduleBody) {
    llvm::outs() << "moduleOp: ";
    llvm::outs() << op << "\n";
    if (auto buffer = dyn_cast<ChannelBuffer>(op))
      toExpand.push_back(buffer);
  }
  for (auto buffer : toExpand)
    expandBuffer(buffer);
};

void ESIToPhysicalPass::expandBuffer(ChannelBuffer buffer) {
  ChannelBufferOptions opts = buffer.options();
  auto type = buffer.getType();

  uint64_t numStages = opts.stages().getUInt();
  Value input = buffer.input();
  auto loc = buffer.getLoc();
  for (size_t i = 0; i < numStages; ++i) {
    auto stage = builder->create<PipelineStage>(loc, type, input);
    input = stage.output();
  }
  buffer.replaceAllUsesWith(input);
}

struct ESIToRTLPass : public LowerESIToRTLBase<ESIToRTLPass> {

  void runOnOperation() override;
};

void ESIToRTLPass::runOnOperation() { llvm::outs() << "test!\n"; }

} // end anonymous namespace.

namespace circt {
namespace esi {
std::unique_ptr<OperationPass<ModuleOp>> createESILoweringPass() {
  return std::make_unique<ESIToPhysicalPass>();
}

std::unique_ptr<OperationPass<ModuleOp>> createESIToRTLPass() {
  return std::make_unique<ESIToRTLPass>();
}
} // namespace esi
} // namespace circt

namespace {
#define GEN_PASS_REGISTRATION
#include "circt/Dialect/ESI/ESIPasses.h.inc"
} // namespace

void circt::esi::registerESIPasses() { registerPasses(); }
