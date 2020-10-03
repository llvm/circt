//===- FIRRTLToLLHD.cpp - FIRRTL to LLHD Conversion Pass ------------------===//
//
// This is the main FIRRTL to LLHD Conversion Pass Implementation.
//
//===----------------------------------------------------------------------===//

#include "circt/Conversion/FIRRTLToLLHD/FIRRTLToLLHD.h"

#include "circt/Dialect/FIRRTL/Ops.h"
#include "circt/Dialect/LLHD/IR/LLHDDialect.h"
#include "circt/Dialect/LLHD/IR/LLHDOps.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Support/Debug.h"
#include "mlir/Pass/Pass.h"

#define DEBUG_TYPE "firrtl-to-llhd"

namespace circt {
namespace llhd {
#define GEN_PASS_CLASSES
#include "circt/Conversion/FIRRTLToLLHD/Passes.h.inc"
}  // namespace llhd
}  // namespace circt

using namespace mlir;
using namespace circt;
using namespace circt::llhd;

namespace {
struct FIRRTLToLLHDPass : public ConvertFIRRTLToLLHDBase<FIRRTLToLLHDPass> {
  void runOnOperation() override;
  void convertModule(firrtl::FModuleOp &module);

 private:
  /// A builder to emit LLHD into.
  OpBuilder *builder = nullptr;
};
}  // namespace

/// Create a FIRRTL to LLHD conversion pass.
std::unique_ptr<OperationPass<firrtl::CircuitOp>>
circt::llhd::createConvertFIRRTLToLLHDPass() {
  return std::make_unique<FIRRTLToLLHDPass>();
}

/// Run the FIRRTL to LLHD conversion pass.
void FIRRTLToLLHDPass::runOnOperation() {
  auto circuit = getOperation();
  LLVM_DEBUG(llvm::dbgs() << "Converting FIRRTL circuit `" << circuit.name()
                          << "` to LLHD\n");

  // Setup a builder which we use to emit LLHD ops.
  OpBuilder theBuilder(&getContext());
  theBuilder.setInsertionPointAfter(circuit);

  // Convert each module separately.
  builder = &theBuilder;
  for (auto &op : circuit.getBody()->getOperations()) {
    if (auto module = dyn_cast<firrtl::FModuleOp>(op)) {
      convertModule(module);
    } else if (!isa<firrtl::DoneOp>(op)) {
      op.emitError("expected `firrtl.module` or `firrtl.done`");
      signalPassFailure();
    }
  }
  builder = nullptr;
}

/// Convert a single FIRRTL module.
void FIRRTLToLLHDPass::convertModule(firrtl::FModuleOp &module) {
  LLVM_DEBUG(llvm::dbgs() << "Converting FIRRTL module `" << module.getName()
                          << "` to LLHD\n");

  // Map the potentially complex FIRRTL module ports to LLHD entity inputs and
  // outputs. This will become fairly involved, since the nested nature of flips
  // and bundle types requires refactoring of the ports.
  SmallVector<firrtl::ModulePortInfo, 4> module_ports;
  module.getPortInfo(module_ports);

  SmallVector<Type, 4> ins;
  SmallVector<Type, 4> outs;
  SmallVector<StringAttr, 4> in_names;
  SmallVector<StringAttr, 4> out_names;
  for (auto &port : llvm::make_early_inc_range(module_ports)) {
    LLVM_DEBUG(llvm::dbgs() << "  - Port " << port.first << " of type "
                            << port.second << "\n");

    // For now, let's do a simple approach where we only support flip at the top
    // of a port's aggregate type.
    bool is_flip = false;
    firrtl::FIRRTLType type = port.second;
    if (auto flip_type = port.second.dyn_cast<firrtl::FlipType>()) {
      is_flip = true;
      type = flip_type.getElementType();
    }

    // Convert the type. We keep things simple for the time being.
    auto width = type.getBitWidthOrSentinel();
    if (width < 0) {
      module.emitError() << "port " << port.first << " has unsupported type "
                         << port.second;
      signalPassFailure();
      continue;
    }
    auto conv_type = SigType::get(builder->getIntegerType(width));

    // Add to the list of inputs or outputs, depending on flip state.
    if (is_flip) {
      outs.push_back(conv_type);
      out_names.push_back(port.first);
    } else {
      ins.push_back(conv_type);
      in_names.push_back(port.first);
    }
  }

  // Concatenate inputs and outputs and mark the split point for the entity.
  // Then assemble the entity signature type.
  auto num_ins = ins.size();
  ins.append(outs.begin(), outs.end());
  in_names.append(out_names.begin(), out_names.end());
  auto entity_type = builder->getFunctionType(ins, llvm::None);

  // Create an LLHD entity for this module.
  auto entity = builder->create<EntityOp>(module.getLoc(), num_ins);
  entity.setName(module.getName());
  entity.setAttr("type", TypeAttr::get(entity_type));
  EntityOp::ensureTerminator(entity.body(), *builder, entity.getLoc());

  // Populate the arguments for the entity.
  entity.body().addArguments(ins);

  // Populate the entity.
  for (auto &op : module.getBodyBlock()->getOperations()) {
    // Skip the dummy terminator.
    if (isa<firrtl::DoneOp>(op)) continue;

    // Unsupported operation.
    op.emitError("conversion to LLHD not supported");
    signalPassFailure();
  }
}

/// Register the FIRRTL to LLHD convesion pass.
namespace {
#define GEN_PASS_REGISTRATION
#include "circt/Conversion/FIRRTLToLLHD/Passes.h.inc"
}  // namespace

void circt::llhd::registerFIRRTLToLLHDPasses() { registerPasses(); }
