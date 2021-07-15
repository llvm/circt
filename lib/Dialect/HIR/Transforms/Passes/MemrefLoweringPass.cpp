//=========- MemrefLoweringPass.cpp - Lower memref type---===//
//
// This file implements lowering pass for memref type.
//
//===----------------------------------------------------------------------===//

#include "../PassDetails.h"
#include "circt/Dialect/HIR/Analysis/FanoutAnalysis.h"
#include "circt/Dialect/HIR/IR/HIR.h"
#include "circt/Dialect/HIR/IR/helper.h"
using namespace circt;
using namespace hir;
namespace {
enum MemrefPortKind { RD = 0, WR = 1, RW = 2 };
struct PortToBusMapping {
  MemrefPortKind portKind;
  Value addrBus;
  Value rdEnableBus;
  Value rdDataBus;
  Value wrEnableBus;
  Value wrDataBus;
};
/// This pass lowers a hir::MemrefType to hir::BusType.
/// RD port becomes three separate tensors of (send
/// addr, send rd_valid, recv rd_data).
/// WR port becomes tensors of (send addr, send wr_valid, send wr_data).
/// RW port becomes tensors of (send addr, send wr_valid, send
/// wr_data, send rd_valid, recv rd_data)
class MemrefLoweringPass : public hir::MemrefLoweringBase<MemrefLoweringPass> {
public:
  void runOnOperation() override;

private:
  LogicalResult visitRegion(Region &);
  LogicalResult dispatch(Operation *);
  LogicalResult visitOp(hir::FuncOp);
  LogicalResult visitOp(hir::AllocaOp);
  LogicalResult visitOp(hir::LoadOp);
  LogicalResult visitOp(hir::StoreOp);
  LogicalResult visitOp(hir::CallOp);

private:
  DenseMap<Value, SmallVector<PortToBusMapping>> mapMemrefPortToBuses;
  llvm::DenseMap<Value, SmallVector<SmallVector<ListOfUses>>> *uses;
};
} // end anonymous namespace

//------------------------------------------------------------------------------
// Helper functions.
//------------------------------------------------------------------------------
mlir::TensorType buildBusTensor(MLIRContext *context, ArrayRef<int64_t> shape,
                                SmallVector<Type> busElementTypes) {
  SmallVector<BusDirection> busDirections;
  // Only same direction is allowed inside a tensor<bus>.
  busDirections.append(busElementTypes.size(), BusDirection::SAME);

  return mlir::RankedTensorType::get(
      shape, hir::BusType::get(context, busElementTypes, busDirections));
}

bool isRead(Attribute port) {
  auto portDict = port.dyn_cast<DictionaryAttr>();
  auto rdLatencyAttr = portDict.getNamed("rd_latency");
  auto wrLatencyAttr = portDict.getNamed("wr_latency");
  assert(rdLatencyAttr || wrLatencyAttr);
  if (rdLatencyAttr)
    return true;
  return false;
}

bool isWrite(Attribute port) {
  auto portDict = port.dyn_cast<DictionaryAttr>();
  auto rdLatencyAttr = portDict.getNamed("rd_latency");
  auto wrLatencyAttr = portDict.getNamed("wr_latency");
  assert(rdLatencyAttr || wrLatencyAttr);
  if (wrLatencyAttr)
    return true;
  return false;
}

void insertBusTypesPerPort(size_t i, Block &bb,
                           SmallVectorImpl<Type> &inputTypes,
                           hir::MemrefType memrefTy, DictionaryAttr port,
                           PortToBusMapping &info) {
  auto *context = memrefTy.getContext();
  Type validTy = buildBusTensor(context, memrefTy.filterShape(BANK),
                                {IntegerType::get(context, 1)});
  Type addrTy = buildBusTensor(
      context, memrefTy.filterShape(BANK),
      {IntegerType::get(context,
                        helper::clog2(memrefTy.getNumElementsPerBank()))});
  Type dataTy = buildBusTensor(context, memrefTy.filterShape(BANK),
                               {memrefTy.getElementType()});
  if (memrefTy.getNumElementsPerBank() > 1) {
    info.addrBus = bb.insertArgument(i, addrTy);
    inputTypes.push_back(addrTy);
  }
  if (isRead(port)) {
    // Enable bus will come first in the list, then data bus.
    info.rdDataBus = bb.insertArgument(i, validTy);
    info.rdEnableBus = bb.insertArgument(i, dataTy);
    inputTypes.push_back(validTy);
    inputTypes.push_back(dataTy);
  }
  if (isWrite(port)) {
    // Enable bus will come first in the list, then data bus.
    info.wrDataBus = bb.insertArgument(i, validTy);
    info.wrEnableBus = bb.insertArgument(i, dataTy);
    inputTypes.push_back(validTy);
    inputTypes.push_back(dataTy);
  }

  if (isRead(port) && isWrite(port))
    info.portKind = RW;
  else if (isRead(port))
    info.portKind = RD;
  else if (isWrite(port))
    info.portKind = WR;
  else
    assert(false);
}

void addBusAttrsPerPort(size_t i, SmallVectorImpl<DictionaryAttr> &attrs,
                        hir::MemrefType memrefTy, DictionaryAttr port) {
  auto *context = memrefTy.getContext();
  if (memrefTy.getNumElementsPerBank() > 1) {
    attrs.push_back(helper::getDictionaryAttr(
        context, "hir.bus.ports",
        ArrayAttr::get(context, StringAttr::get(context, "send"))));
  }
  if (isRead(port)) {
    attrs.push_back(helper::getDictionaryAttr(
        context, "hir.bus.ports",
        ArrayAttr::get(context, StringAttr::get(context, "send"))));
    attrs.push_back(helper::getDictionaryAttr(
        context, "hir.bus.ports",
        ArrayAttr::get(context, StringAttr::get(context, "recv"))));
  }
  if (isWrite(port)) {
    attrs.push_back(helper::getDictionaryAttr(
        context, "hir.bus.ports",
        ArrayAttr::get(context, StringAttr::get(context, "send"))));
    attrs.push_back(helper::getDictionaryAttr(
        context, "hir.bus.ports",
        ArrayAttr::get(context, StringAttr::get(context, "send"))));
  }
}

void insertBusArguments(
    hir::FuncOp op,
    DenseMap<Value, SmallVector<PortToBusMapping>> &mapMemrefPortToBuses) {
  SmallVector<Type> inputTypes;
  SmallVector<DictionaryAttr> inputAttrs;
  auto funcTy = op.funcTy().dyn_cast<hir::FuncType>();
  auto &bb = op.getFuncBody().front();
  for (int i = bb.getNumArguments() - 1; i >= 0; i--) {
    Value arg = bb.getArgument(i);
    if (auto memrefTy = arg.getType().dyn_cast<hir::MemrefType>()) {
      auto ports = helper::extractMemrefPortsFromDict(inputAttrs[i]);
      for (auto port : ports) {
        auto portDict = port.dyn_cast<DictionaryAttr>();
        assert(portDict);
        PortToBusMapping info;
        insertBusTypesPerPort(i, bb, inputTypes, memrefTy, portDict, info);
        addBusAttrsPerPort(i, inputAttrs, memrefTy, portDict);
      }
    }
    inputTypes.push_back(funcTy.getInputTypes()[i]);
    inputAttrs.push_back(funcTy.getInputAttrs()[i]);
  }
  auto newFuncTy =
      hir::FuncType::get(op.getContext(), inputTypes, inputAttrs,
                         funcTy.getResultTypes(), funcTy.getResultAttrs());
  op.funcTyAttr(TypeAttr::get(newFuncTy));
}

void removeMemrefArguments(hir::FuncOp op) {
  auto *context = op.getContext();
  auto funcTy = op.funcTy().dyn_cast<hir::FuncType>();
  SmallVector<Type> inputTypes;
  SmallVector<DictionaryAttr> inputAttrs;
  auto &bb = op.getFuncBody().front();
  for (int i = bb.getNumArguments() - 1; i >= 0; i--) {
    if (bb.getArgument(i).getType().isa<hir::MemrefType>()) {
      bb.eraseArgument(i);
    } else {
      inputTypes.push_back(funcTy.getInputTypes()[i]);
      inputAttrs.push_back(funcTy.getInputAttrs()[i]);
    }
  }
  auto newFuncTy =
      hir::FuncType::get(context, inputTypes, inputAttrs,
                         funcTy.getResultTypes(), funcTy.getResultAttrs());
  op.funcTyAttr(TypeAttr::get(newFuncTy));
}

void initBusesForNoMemrefUse(uint64_t bank, PortToBusMapping portToBusMapping) {
  // TODO
}
PortToBusMapping defineRdBuses(uint64_t bank, uint64_t numUses,
                               PortToBusMapping portToBusMapping) {
  PortToBusMapping buses;
  // TODO
  return buses;
}

void connectRdBuses(uint64_t bank, PortToBusMapping rdBusesPerBank,
                    PortToBusMapping buses) {
  // TODO
}

llvm::Optional<PortToBusMapping>
defineBusesForMemrefUse(uint64_t bank, ListOfUses uses,
                        PortToBusMapping portToBusMapping) {
  auto portKind = portToBusMapping.portKind;
  if (uses.size() == 0) {
    initBusesForNoMemrefUse(bank, portToBusMapping);
    return llvm::None;
  }
  if (portKind == RD) {
    auto rdBuses = defineRdBuses(bank, uses.size(), portToBusMapping);
    connectRdBuses(bank, rdBuses, portToBusMapping);
    return rdBuses;
  }
  if (portKind == WR) {
    auto wrBuses = defineWrBuses(bank, uses.size(), portToBusMapping);
    connectWrBuses(bank, wrBuses, portToBusMapping);
    return wrBuses;
  }
  auto rwBuses = defineRWBuses(bank, uses.size(), portToBusMapping);
  connectRWBuses(bank, rwBuses, portToBusMapping);
  return rwBuses;
}

LogicalResult replacePortUses(PortToBusMapping portToBusMapping,
                              SmallVector<ListOfUses> portUses) {
  uint64_t numBanks = portUses.size();
  for (uint64_t bank = 0; bank < numBanks; bank++) {
    auto buses =
        defineBusesForMemrefUse(bank, portUses[bank], portToBusMapping);
    for (uint64_t use = 0; use < portUses[bank].size(); use++) {
      if (failed(convertLoadOp(portUses[bank][use], buses, use)))
        return failure();
    }
  }
  return success();
}

LogicalResult replaceMemrefUses(
    hir::FuncOp op,
    DenseMap<Value, SmallVector<PortToBusMapping>> mapMemrefPortToBuses,
    DenseMap<Value, SmallVector<SmallVector<ListOfUses>>> &uses) {
  OpBuilder builder(op);
  builder.setInsertionPointToStart(&op.getFuncBody().front());
  // Create a list of memref input args.
  SmallVector<Value> memrefArgs;
  for (auto arg : op.getFuncBody().front().getArguments())
    if (arg.getType().isa<hir::MemrefType>())
      memrefArgs.push_back(arg);

  // For each memref-port-bank define new tensor<bus> depending on the number
  // of uses and portKind.
  for (auto memref : memrefArgs) {
    SmallVector<PortToBusMapping> &portsToBusMapping =
        mapMemrefPortToBuses[memref];
    SmallVector<SmallVector<ListOfUses>> memrefUses = uses[memref];
    for (uint64_t port = 0; port < portsToBusMapping.size(); port++) {
      PortToBusMapping portToBusMapping = portsToBusMapping[port];
      SmallVector<ListOfUses> portUses = memrefUses[port];
      if (failed(replacePortUses(portToBusMapping, portUses)))
        return failure();
    }
  }
}
//------------------------------------------------------------------------------

//------------------------------------------------------------------------------
// MemrefLoweringPass methods.
//------------------------------------------------------------------------------
LogicalResult MemrefLoweringPass::visitOp(hir::FuncOp op) {
  insertBusArguments(op, mapMemrefPortToBuses);

  if (failed(visitRegion(op.getFuncBody())))
    return failure();

  removeMemrefArguments(op);
  if (failed(replaceMemrefUses(op, mapMemrefPortToBuses, *uses)))
    return failure();
  return success();
}

LogicalResult MemrefLoweringPass::visitRegion(Region &region) {
  assert(region.getBlocks().size() == 1);
  for (auto &operation : region.front()) {
    if (auto op = dyn_cast<hir::FuncOp>(operation)) {
      if (failed(visitOp(op)))
        return failure();
    } else if (auto op = dyn_cast<hir::AllocaOp>(operation)) {
      if (failed(visitOp(op)))
        return failure();
    } else if (auto op = dyn_cast<hir::LoadOp>(operation)) {
      if (failed(visitOp(op)))
        return failure();
    } else if (auto op = dyn_cast<hir::StoreOp>(operation)) {
      if (failed(visitOp(op)))
        return failure();
    } else if (auto op = dyn_cast<hir::CallOp>(operation)) {
      return op.emitError("MemrefLoweringPass does not support CallOp. "
                          "Inline the functions.");
    }
  }
  return success();
}

void MemrefLoweringPass::runOnOperation() {
  hir::FuncOp funcOp = getOperation();
  auto memrefUseInfo = MemrefUseInfo(funcOp);
  this->uses = &memrefUseInfo.mapMemref2PerPortPerBankUses;
  if (failed(visitOp(funcOp))) {
    signalPassFailure();
    return;
  }
}

namespace circt {
namespace hir {
std::unique_ptr<OperationPass<hir::FuncOp>> createMemrefLoweringPass() {
  return std::make_unique<MemrefLoweringPass>();
}
} // namespace hir
} // namespace circt
