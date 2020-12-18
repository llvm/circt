//===- RTLToLLHD.cpp - RTL to LLHD Conversion Pass ------------------------===//
//
// This is the main RTL to LLHD Conversion Pass Implementation.
//
//===----------------------------------------------------------------------===//

#include "circt/Conversion/RTLToLLHD/RTLToLLHD.h"
#include "circt/Dialect/LLHD/IR/LLHDDialect.h"
#include "circt/Dialect/LLHD/IR/LLHDOps.h"
#include "circt/Dialect/RTL/Ops.h"
#include "circt/Dialect/RTL/Visitors.h"
#include "circt/Support/ImplicitLocOpBuilder.h"
#include "mlir/IR/StandardTypes.h"
#include "mlir/Pass/Pass.h"

namespace circt {
namespace llhd {
#define GEN_PASS_CLASSES
#include "circt/Conversion/RTLToLLHD/Passes.h.inc"
} // namespace llhd
} // namespace circt

using namespace circt;
using namespace llhd;

//===----------------------------------------------------------------------===//
// RTL to LLHD Conversion Pass
//===----------------------------------------------------------------------===//

namespace {
struct RTLToLLHDPass
    : public ConvertRTLToLLHDBase<RTLToLLHDPass>,
      public rtl::CombinatorialVisitor<RTLToLLHDPass, LogicalResult>,
      public rtl::StmtVisitor<RTLToLLHDPass, LogicalResult> {
  void runOnOperation() override;
  LogicalResult visitModule(rtl::RTLModuleOp module);

private:
  ImplicitLocOpBuilder *builder = nullptr;
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

/// This is the main entrypoint for the RTL to LLHD conversion pass.
void RTLToLLHDPass::runOnOperation() {
  mlir::ModuleOp module = getOperation();

  // Instantiate the builder to use for this module.
  ImplicitLocOpBuilder theBuilder(module.getLoc(), &getContext());
  builder = &theBuilder;

  // Walk the RTL modules and attempt to convert them.
  WalkResult result = module.walk([&](rtl::RTLModuleOp rtlModule) {
    builder->setInsertionPointAfter(rtlModule);
    return WalkResult(visitModule(rtlModule));
  });

  // If anything went wrong, signal failure.
  if (result.wasInterrupted())
    signalPassFailure();
}

//===----------------------------------------------------------------------===//
// Convert modules.
//===----------------------------------------------------------------------===//

/// This works on each RTL module, creates corresponding entities, moves the
/// bodies of the modules into the entities, and converts the bodies.
LogicalResult RTLToLLHDPass::visitModule(rtl::RTLModuleOp module) {
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

  // Create the entity, set its type and name, and create its body terminator.
  auto entity = builder->create<EntityOp>(entityInputs.size());
  entityInputs.append(entityOutputs.begin(), entityOutputs.end());
  auto entityType = builder->getFunctionType(entityInputs, {});
  entity.setAttr(entity.getTypeAttrName(), TypeAttr::get(entityType));
  entity.setName(module.getName());
  EntityOp::ensureTerminator(entity.body(), *builder, entity.getLoc());

  // Add the entity block arguments.
  entity.body().addArguments(entityInputs);

  // Erase the RTL module.
  module.erase();

  return success();
}
