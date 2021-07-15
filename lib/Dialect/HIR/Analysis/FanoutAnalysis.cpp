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
  WalkResult result = funcOp.walk([this](Operation *operation) -> WalkResult {
    if (auto op = dyn_cast<hir::LoadOp>(operation))
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

  int numExtraPortsNeeded = port + 1 - mapMemref2PerPortPerBankUses[mem].size();
  if (numExtraPortsNeeded > 0)
    mapMemref2PerPortPerBankUses[mem].append(numExtraPortsNeeded,
                                             SmallVector<ListOfUses>());
  if (mapMemref2PerPortPerBankUses[mem][port].empty())
    mapMemref2PerPortPerBankUses[mem][port].append(memrefTy.getNumBanks(),
                                                   ListOfUses());
  assert(mapMemref2PerPortPerBankUses[mem][port].size() ==
         memrefTy.getNumBanks());
  mapMemref2PerPortPerBankUses[mem][port][bank].push_back(op);
}

void MemrefUseInfo::visitOp(hir::LoadOp op) {
  Value mem = op.mem();
  auto memrefTy = mem.getType().dyn_cast<hir::MemrefType>();
  assert(mapMemref2PerPortPerBankUses.find(mem) !=
         mapMemref2PerPortPerBankUses.end());
  int64_t linearIdx = helper::calcLinearIndex(op.filerIndices(BANK),
                                              memrefTy.filterShape(BANK));
  // before lowering explicit port must be specified.
  assert(op.port().hasValue());
  uint64_t port = op.port().getValue();
  addOpToUseList(op, mem, port, linearIdx);
}

void MemrefUseInfo::visitOp(hir::StoreOp op) {
  Value mem = op.mem();
  auto memrefTy = mem.getType().dyn_cast<hir::MemrefType>();
  assert(mapMemref2PerPortPerBankUses.find(mem) !=
         mapMemref2PerPortPerBankUses.end());
  int64_t linearIdx = helper::calcLinearIndex(op.filerIndices(BANK),
                                              memrefTy.filterShape(BANK));
  // before lowering explicit port must be specified.
  assert(op.port().hasValue());
  auto port = op.port().getValue();
  addOpToUseList(op, mem, port, linearIdx);
}
