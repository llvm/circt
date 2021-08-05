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
  visitOp(funcOp);
  funcOp.walk([this](Operation *operation) -> WalkResult {
    if (auto op = dyn_cast<hir::AllocaOp>(operation))
      visitOp(op);
    else if (auto op = dyn_cast<hir::LoadOp>(operation))
      visitOp(op);
    else if (auto op = dyn_cast<hir::StoreOp>(operation))
      visitOp(op);
    else if (auto op = dyn_cast<hir::CallOp>(operation))
      assert(false); // CallOp not implemented yet.
    return WalkResult::advance();
  });
}

void MemrefUseInfo::addOpToUseList(Operation *op, Value mem, uint64_t port,
                                   uint64_t bank) {
  hir::MemrefType memrefTy = mem.getType().dyn_cast<hir::MemrefType>();

  assert(memrefTy);
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
}

void MemrefUseInfo::visitOp(hir::FuncOp op) {
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
}

void MemrefUseInfo::visitOp(hir::AllocaOp op) {
  auto memrefTy = op.res().getType().dyn_cast<hir::MemrefType>();
  assert(memrefTy);
  auto ports = op.ports();

  SmallVector<ListOfUses> listOfUsesPerBank;
  listOfUsesPerBank.append(memrefTy.getNumBanks(), ListOfUses());
  mapMemref2PerPortPerBankUses[op.res()].append(ports.size(),
                                                listOfUsesPerBank);
}

void MemrefUseInfo::visitOp(hir::LoadOp op) {
  Value mem = op.mem();
  auto memrefTy = mem.getType().dyn_cast<hir::MemrefType>();
  int64_t linearIdx = helper::calcLinearIndex(op.filterIndices(BANK),
                                              memrefTy.filterShape(BANK));
  // before lowering explicit port must be specified.
  assert(op.port().hasValue());
  uint64_t port = op.port().getValue();
  addOpToUseList(op, mem, port, linearIdx);
}

void MemrefUseInfo::visitOp(hir::StoreOp op) {
  Value mem = op.mem();
  auto memrefTy = mem.getType().dyn_cast<hir::MemrefType>();
  int64_t linearIdx = helper::calcLinearIndex(op.filterIndices(BANK),
                                              memrefTy.filterShape(BANK));
  // before lowering explicit port must be specified.
  assert(op.port().hasValue());
  auto port = op.port().getValue();
  addOpToUseList(op, mem, port, linearIdx);
}
