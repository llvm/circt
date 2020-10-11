//===- FIRRTLToLLHD.cpp - FIRRTL to LLHD Conversion Pass ------------------===//
//
// This is the main FIRRTL to LLHD Conversion Pass Implementation.
//
//===----------------------------------------------------------------------===//

#include "circt/Conversion/FIRRTLToLLHD/FIRRTLToLLHD.h"

#include "circt/Dialect/FIRRTL/Ops.h"
#include "circt/Dialect/FIRRTL/Visitors.h"
#include "circt/Dialect/LLHD/IR/LLHDDialect.h"
#include "circt/Dialect/LLHD/IR/LLHDOps.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "firrtl-to-llhd"

namespace circt {
namespace llhd {
#define GEN_PASS_CLASSES
#include "circt/Conversion/FIRRTLToLLHD/Passes.h.inc"
} // namespace llhd
} // namespace circt

using namespace mlir;
using namespace circt;
using namespace circt::llhd;

namespace {
struct FIRRTLToLLHDPass
    : public ConvertFIRRTLToLLHDBase<FIRRTLToLLHDPass>,
      public firrtl::FIRRTLVisitor<FIRRTLToLLHDPass, LogicalResult> {
  void runOnOperation() override;
  void convertCircuit(firrtl::CircuitOp &module);
  void convertModule(firrtl::FModuleOp &module);

  // FIRRTL Visitor
  using firrtl::FIRRTLVisitor<FIRRTLToLLHDPass, LogicalResult>::visitExpr;
  using firrtl::FIRRTLVisitor<FIRRTLToLLHDPass, LogicalResult>::visitDecl;
  using firrtl::FIRRTLVisitor<FIRRTLToLLHDPass, LogicalResult>::visitStmt;

  LogicalResult visitUnhandledOp(Operation *op);
  LogicalResult visitInvalidOp(Operation *op);

  LogicalResult visitStmt(firrtl::DoneOp op) { return success(); }
  LogicalResult visitStmt(firrtl::ConnectOp op);

private:
  /// A builder to emit LLHD into.
  OpBuilder *builder = nullptr;

  // Conversion helpers
  DenseMap<Value, Value> valueMapping;
  Value getConvertedValue(Value operand);
  Value getConvertedAndExtendedValue(Value operand, Type destType);
};
} // namespace

/// Create a FIRRTL to LLHD conversion pass.
std::unique_ptr<OperationPass<ModuleOp>>
circt::llhd::createConvertFIRRTLToLLHDPass() {
  return std::make_unique<FIRRTLToLLHDPass>();
}

/// Run the FIRRTL to LLHD conversion pass.
void FIRRTLToLLHDPass::runOnOperation() {
  for (auto &op :
       llvm::make_early_inc_range(getOperation().getBody()->getOperations())) {
    if (auto circuit = dyn_cast<firrtl::CircuitOp>(op)) {
      convertCircuit(circuit);
      circuit.getOperation()->dropAllReferences();
      circuit.getOperation()->dropAllDefinedValueUses();
      circuit.getOperation()->erase();
    }
  }
}

/// Convert an entire FIRRTL circuit.
void FIRRTLToLLHDPass::convertCircuit(firrtl::CircuitOp &circuit) {
  LLVM_DEBUG(llvm::dbgs() << "Converting FIRRTL circuit `" << circuit.name()
                          << "` to LLHD\n");

  // Setup a builder which we use to emit LLHD ops.
  OpBuilder theBuilder(&getContext());

  // Convert each module separately.
  builder = &theBuilder;
  for (auto &op : circuit.getBody()->getOperations()) {
    if (auto module = dyn_cast<firrtl::FModuleOp>(op)) {
      builder->setInsertionPointAfter(circuit);
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
  SmallVector<firrtl::ModulePortInfo, 4> modulePorts;
  module.getPortInfo(modulePorts);

  SmallVector<Type, 4> ins;
  SmallVector<Type, 4> outs;
  SmallVector<StringAttr, 4> inNames;
  SmallVector<StringAttr, 4> outNames;
  SmallVector<unsigned, 4> inIndices;
  SmallVector<unsigned, 4> outIndices;
  for (unsigned i = 0; i < modulePorts.size(); i++) {
    auto &port = modulePorts[i];
    LLVM_DEBUG(llvm::dbgs()
               << "Port " << port.first << " of type " << port.second << "\n");

    // For now, let's do a simple approach where we only support flip at the top
    // of a port's aggregate type.
    bool isFlip = false;
    firrtl::FIRRTLType type = port.second;
    if (auto flipType = port.second.dyn_cast<firrtl::FlipType>()) {
      isFlip = true;
      type = flipType.getElementType();
    }

    // Convert the type. We keep things simple for the time being.
    auto width = type.getBitWidthOrSentinel();
    if (width < 0) {
      module.emitError() << "port " << port.first << " has unsupported type "
                         << port.second;
      signalPassFailure();
      continue;
    }
    auto convType = SigType::get(builder->getIntegerType(width));

    // Add to the list of inputs or outputs, depending on flip state.
    if (isFlip) {
      outs.push_back(convType);
      outNames.push_back(port.first);
      outIndices.push_back(i);
    } else {
      ins.push_back(convType);
      inNames.push_back(port.first);
      inIndices.push_back(i);
    }
  }

  // Concatenate inputs and outputs and mark the split point for the entity.
  // Then assemble the entity signature type.
  auto numIns = ins.size();
  ins.append(outs.begin(), outs.end());
  inNames.append(outNames.begin(), outNames.end());
  inIndices.append(outIndices.begin(), outIndices.end());
  auto entityType = builder->getFunctionType(ins, llvm::None);

  // Create an LLHD entity for this module.
  auto entity = builder->create<EntityOp>(module.getLoc(), numIns);
  entity.setName(module.getName());
  entity.setAttr("type", TypeAttr::get(entityType));
  EntityOp::ensureTerminator(entity.body(), *builder, entity.getLoc());

  // Populate the arguments for the entity. This includes initializing the value
  // mapping table with correspondences of the FIRRTL module arguments to the
  // LLHD entity inputs/outputs. These are likely to be in a different order,
  // such that we use the indirection indices gathered above.
  auto args = entity.body().addArguments(ins);
  for (auto arg : llvm::zip(args, inIndices)) {
    auto firrtlArg = module.getBodyBlock()->getArgument(std::get<1>(arg));
    auto llhdArg = std::get<0>(arg);
    LLVM_DEBUG(llvm::dbgs()
               << "Map FIRRTL port " << firrtlArg.getArgNumber()
               << " to LLHD port " << llhdArg.getArgNumber() << "\n");
    valueMapping[firrtlArg] = llhdArg;
  }

  // Populate the entity.
  builder->setInsertionPoint(entity.body().front().getTerminator());
  for (auto &op : module.getBodyBlock()->getOperations()) {
    if (failed(dispatchVisitor(&op)))
      signalPassFailure();
  }
}

//===----------------------------------------------------------------------===//
// Conversion Helpers
//===----------------------------------------------------------------------===//

/// Get the LLHD value that has been previously emitted for a FIRRTL operand.
Value FIRRTLToLLHDPass::getConvertedValue(Value operand) {
  auto it = valueMapping.find(operand);
  assert(it != valueMapping.end() && "operand has not been converted to LLHD");
  return it->second;
}

/// Get the LLHD value that has been previously emitted for a FIRRTL operand,
/// and implicitly extend its type as appropriate.
Value FIRRTLToLLHDPass::getConvertedAndExtendedValue(Value operand,
                                                     Type destType) {
  if (operand.getType().cast<firrtl::FIRRTLType>().getPassiveType() !=
      destType.cast<firrtl::FIRRTLType>().getPassiveType()) {
    emitError(operand.getLoc(), "implicit extension not implemented");
    return {};
  }
  return getConvertedValue(operand);
}

//===----------------------------------------------------------------------===//
// Statements
//===----------------------------------------------------------------------===//

LogicalResult FIRRTLToLLHDPass::visitUnhandledOp(Operation *op) {
  op->emitError() << "conversion to LLHD not supported";
  return failure();
}

LogicalResult FIRRTLToLLHDPass::visitInvalidOp(Operation *op) {
  op->emitError() << "conversion to LLHD not supported";
  return failure();
}

LogicalResult FIRRTLToLLHDPass::visitStmt(firrtl::ConnectOp op) {
  LLVM_DEBUG(llvm::dbgs() << "Converting " << op << "\n");

  auto dst = getConvertedValue(op.lhs());
  auto src = getConvertedAndExtendedValue(op.rhs(), op.lhs().getType());
  if (!dst || !src)
    return failure();

  LLVM_DEBUG(llvm::dbgs() << "Assigning " << src.getType() << " to "
                          << dst.getType() << "\n");

  // We must be connecting to a signal.
  assert(dst.getType().isa<SigType>() && "must connect to signal");

  // If we are connecting two signals, e.g. an input to an output port, use an
  // `llhd.con` operation. Otherwise use `llhd.drv` to drive the RHS onto the
  // signal of the LHS.
  if (auto sigTy = src.getType().dyn_cast<SigType>()) {
    src = builder->create<PrbOp>(op.getLoc(), sigTy.getUnderlyingType(), src);
  }

  // Construct the `1d` time value for the drive.
  auto timeType = TimeType::get(&getContext());
  auto deltaAttr =
      TimeAttr::get(timeType, std::array<unsigned, 3>{0, 1, 0}, "s");
  auto delta = builder->create<ConstOp>(op.getLoc(), timeType, deltaAttr);

  // Construct a constant one for the drive condition.
  auto boolType = builder->getIntegerType(1);
  auto constOneAttr = IntegerAttr::get(boolType, 1);
  auto constOne = builder->create<ConstOp>(op.getLoc(), boolType, constOneAttr);

  // Emit the drive operation.
  builder->create<DrvOp>(op.getLoc(), dst, src, delta, constOne);
  return success();
}

/// Register the FIRRTL to LLHD convesion pass.
namespace {
#define GEN_PASS_REGISTRATION
#include "circt/Conversion/FIRRTLToLLHD/Passes.h.inc"
} // namespace

void circt::llhd::registerFIRRTLToLLHDPasses() { registerPasses(); }
