//=========- MemrefLoweringPass.cpp - Lower memref type---===//
//
// This file implements lowering pass for memref type.
//
//===----------------------------------------------------------------------===//

#include "../PassDetails.h"
#include "circt/Dialect/HIR/IR/HIR.h"
#include "circt/Dialect/HIR/IR/helper.h"

using namespace circt;
using namespace hir;
namespace {
struct Info {
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
  DenseMap<Value, SmallVector<Info>> infoPerMemrefPerPort;
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
                           Info &info) {
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
    hir::FuncOp op, DenseMap<Value, SmallVector<Info>> &infoPerMemrefPerPort) {
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
        Info info;
        insertBusTypesPerPort(i, bb, inputTypes, memrefTy, portDict, info);
        addBusAttrsPerPort(i, inputAttrs, memrefTy, portDict);
        infoPerMemrefPerPort[arg].push_back(info);
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
//------------------------------------------------------------------------------

//------------------------------------------------------------------------------
// MemrefLoweringPass methods.
//------------------------------------------------------------------------------
LogicalResult MemrefLoweringPass::visitOp(hir::FuncOp op) {
  insertBusArguments(op, infoPerMemrefPerPort);

  if (failed(visitRegion(op.getFuncBody())))
    return failure();

  removeMemrefArguments(op);
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
      if (failed(visitOp(op)))
        return failure();
    }
  }
  return success();
}

void MemrefLoweringPass::runOnOperation() {
  hir::FuncOp funcOp = getOperation();
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
