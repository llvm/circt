//=========- -------MemrefFanoutInfo.cpp - Analysis pass for bus
// fanout-------===//
//
// Calculates fanout info of each bus in a FuncOp.
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/HIR/Analysis/FanoutAnalysis.h"
#include "circt/Dialect/HIR/IR/HIR.h"
#include "circt/Dialect/HIR/IR/helper.h"

using namespace circt;
using namespace hir;

MemrefUseInfo::MemrefUseInfo(FuncOp funcOp) {
  if (failed(visitOp(funcOp)))
    return;

  funcOp.walk([this](Operation *operation) -> WalkResult {
    LogicalResult visitResult = success();
    if (auto op = dyn_cast<hir::CallOp>(operation))
      visitResult = visitOp(op);
    else if (auto op = dyn_cast<hir::AllocaOp>(operation))
      visitResult = visitOp(op);
    else if (auto op = dyn_cast<hir::LoadOp>(operation))
      visitResult = visitOp(op);
    else if (auto op = dyn_cast<hir::StoreOp>(operation))
      visitResult = visitOp(op);

    if (failed(visitResult))
      return WalkResult::interrupt();
    return WalkResult::advance();
  });
}

LogicalResult MemrefUseInfo::addOpToUseList(Operation *op, Value mem,
                                            uint64_t port, uint64_t bank) {
  hir::MemrefType memrefTy = mem.getType().dyn_cast<hir::MemrefType>();

  assert(memrefTy);
  assert(mapMemref2PerPortPerBankUses.find(mem) !=
         mapMemref2PerPortPerBankUses.end());
  if (mapMemref2PerPortPerBankUses[mem].size() <= port) {
    op->emitError() << "mapMemref2PerPortPerBankUses[mem].size() = "
                    << mapMemref2PerPortPerBankUses[mem].size()
                    << ", port=" << port;
    assert(mapMemref2PerPortPerBankUses[mem].size() > port);
  }

  assert(mapMemref2PerPortPerBankUses[mem][port].size() ==
         memrefTy.getNumBanks());
  assert(memrefTy.getNumBanks() > bank);
  mapMemref2PerPortPerBankUses[mem][port][bank].push_back(op);
  return success();
}

LogicalResult MemrefUseInfo::visitOp(hir::FuncOp op) {
  auto &bb = op.getFuncBody().front();
  auto inputAttrs = op.getFuncType().getInputAttrs();
  for (uint64_t i = 0; i < bb.getNumArguments() - 1; i++) {
    auto arg = bb.getArgument(i);
    if (auto memrefTy = arg.getType().dyn_cast<MemrefType>()) {
      auto ports = helper::extractMemrefPortsFromDict(inputAttrs[i]);
      assert(ports);
      SmallVector<ListOfUses> listOfUsesPerBank;
      listOfUsesPerBank.append(memrefTy.getNumBanks(), ListOfUses());
      mapMemref2PerPortPerBankUses[arg].append(ports->size(),
                                               listOfUsesPerBank);
    }
  }
  return success();
}

LogicalResult MemrefUseInfo::visitOp(hir::CallOp op) {
  for (auto arg : op.operands()) {
    if (arg.getType().isa<hir::MemrefType>())
      return op.emitError() << "FanoutAnalysis pass expects all hir.call with "
                               "memref parameters to be inlined.";
  }
  return success();
}

LogicalResult MemrefUseInfo::visitOp(hir::AllocaOp op) {
  auto memrefTy = op.res().getType().dyn_cast<hir::MemrefType>();
  assert(memrefTy);
  auto ports = op.ports();

  SmallVector<ListOfUses> listOfUsesPerBank;
  listOfUsesPerBank.append(memrefTy.getNumBanks(), ListOfUses());
  mapMemref2PerPortPerBankUses[op.res()].append(ports.size(),
                                                listOfUsesPerBank);
  return success();
}

LogicalResult MemrefUseInfo::visitOp(hir::LoadOp op) {
  Value mem = op.mem();
  auto memrefTy = mem.getType().dyn_cast<hir::MemrefType>();
  auto linearIdx = helper::calcLinearIndex(op.filterIndices(BANK),
                                           memrefTy.filterShape(BANK));
  if (!linearIdx)
    return op.emitError("Could not calculate the bank index. Check if all bank "
                        "indices are constants.");

  // before lowering explicit port must be specified.
  assert(op.port().hasValue());
  uint64_t port = op.port().getValue();
  return addOpToUseList(op, mem, port, linearIdx.getValue());
}

LogicalResult MemrefUseInfo::visitOp(hir::StoreOp op) {
  Value mem = op.mem();
  auto memrefTy = mem.getType().dyn_cast<hir::MemrefType>();
  auto linearIdx = helper::calcLinearIndex(op.filterIndices(BANK),
                                           memrefTy.filterShape(BANK));
  if (!linearIdx)
    return op.emitError("Could not calculate the bank index. Check if all bank "
                        "indices are constants.");

  // before lowering explicit port must be specified.
  assert(op.port().hasValue());
  auto port = op.port().getValue();
  return addOpToUseList(op, mem, port, linearIdx.getValue());
}
