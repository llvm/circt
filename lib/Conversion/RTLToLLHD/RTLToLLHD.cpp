//===- RTLToLLHD.cpp - RTL to LLHD Conversion Pass ------------------------===//
//
// This is the main RTL to LLHD Conversion Pass Implementation.
//
//===----------------------------------------------------------------------===//

#include "circt/Conversion/RTLToLLHD/RTLToLLHD.h"
#include "circt/Dialect/LLHD/IR/LLHDDialect.h"
#include "circt/Dialect/LLHD/IR/LLHDOps.h"
#include "circt/Dialect/RTL/Dialect.h"
#include "circt/Dialect/RTL/Ops.h"
#include "mlir/IR/StandardTypes.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"

namespace circt {
namespace llhd {
#define GEN_PASS_CLASSES
#include "circt/Conversion/RTLToLLHD/Passes.h.inc"
} // namespace llhd
} // namespace circt

using namespace circt;
using namespace llhd;
using namespace rtl;

//===----------------------------------------------------------------------===//
// RTL to LLHD Conversion Pass
//===----------------------------------------------------------------------===//

namespace {
struct RTLToLLHDPass : public ConvertRTLToLLHDBase<RTLToLLHDPass> {
  void runOnOperation() override;
};
} // namespace

/// Create a RTL to LLHD conversion pass.
std::unique_ptr<OperationPass<mlir::ModuleOp>>
llhd::createConvertRTLToLLHDPass() {
  return std::make_unique<RTLToLLHDPass>();
}

/// Register the RTL to LLHD conversion pass.
namespace {
#define GEN_PASS_REGISTRATION
#include "circt/Conversion/RTLToLLHD/Passes.h.inc"
} // namespace

void circt::llhd::registerRTLToLLHDPasses() { registerPasses(); }

/// Forward declare conversion patterns.
struct ConvertRTLModule;

/// This is the main entrypoint for the RTL to LLHD conversion pass.
void RTLToLLHDPass::runOnOperation() {
  MLIRContext &context = getContext();
  ModuleOp module = getOperation();

  ConversionTarget target(context);
  target.addLegalDialect<LLHDDialect>();
  target.addIllegalOp<RTLModuleOp>();

  OwningRewritePatternList patterns;
  patterns.insert<ConvertRTLModule>(&context);

  if (failed(applyPartialConversion(module, target, std::move(patterns))))
    signalPassFailure();
}

//===----------------------------------------------------------------------===//
// Convert modules.
//===----------------------------------------------------------------------===//

/// This works on each RTL module, creates corresponding entities, moves the
/// bodies of the modules into the entities, and converts the bodies.
struct ConvertRTLModule : public OpConversionPattern<RTLModuleOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(RTLModuleOp module, ArrayRef<Value> operands,
                                ConversionPatternRewriter &rewriter) const {
    // Collect module port info.
    SmallVector<rtl::ModulePortInfo, 4> modulePorts;
    rtl::getModulePortInfo(module, modulePorts);

    // Collect the new entity's argument types.
    SmallVector<Type, 4> entityInputs;
    SmallVector<Type, 8> entityOutputs;
    for (auto &port : modulePorts) {
      if (!port.type.isa<IntegerType>())
        return module.emitError("ports with type ")
               << port.type << " are not supported";
      switch (port.direction) {

      case rtl::PortDirection::INPUT:
        entityInputs.push_back(SigType::get(port.type));
        break;

      case rtl::PortDirection::OUTPUT:
        entityOutputs.push_back(SigType::get(port.type));
        break;

      case rtl::PortDirection::INOUT:
        return module.emitError("ports with direction inout are not supported");
      }
    }

    // Create the entity, set its type and name, and create its body
    // terminator.
    auto entity =
        rewriter.create<EntityOp>(module.getLoc(), entityInputs.size());
    entityInputs.append(entityOutputs.begin(), entityOutputs.end());
    auto entityType = rewriter.getFunctionType(entityInputs, {});
    entity.setAttr(entity.getTypeAttrName(), TypeAttr::get(entityType));
    entity.setName(module.getName());
    EntityOp::ensureTerminator(entity.body(), rewriter, entity.getLoc());

    // Add the entity block arguments.
    // TODO: use the rewriter to actually move the body over.
    entity.body().addArguments(entityInputs);

    // Erase the RTL module.
    rewriter.eraseOp(module);

    return success();
  }
};
