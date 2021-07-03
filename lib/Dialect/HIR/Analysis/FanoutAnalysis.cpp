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

MemrefFanoutInfo::MemrefFanoutInfo(Operation *op) {
  auto funcOp = dyn_cast<hir::FuncOp>(op);
  assert(funcOp);

  visitOp(funcOp);

  WalkResult result = funcOp.walk([this](Operation *operation) -> WalkResult {
    if (auto op = dyn_cast<hir::AllocaOp>(operation))
      visitOp(op);
    else if (auto op = dyn_cast<hir::LoadOp>(operation))
      visitOp(op);
    else if (auto op = dyn_cast<hir::StoreOp>(operation))
      visitOp(op);
    else if (auto op = dyn_cast<hir::CallOp>(operation))
      visitOp(op);
    return WalkResult::advance();
  });
  assert(!result.wasInterrupted());
}

void MemrefFanoutInfo::visitOp(hir::FuncOp op) {
  auto &entryBlock = op.getFuncBody();
  auto arguments = entryBlock.getArguments();
  ArrayRef<DictionaryAttr> inputAttrs =
      op.funcTy().dyn_cast<hir::FuncType>().getInputAttrs();
  for (size_t i = 0; i < arguments.size(); i++) {
    auto arg = arguments[i];
    // initialize the map with a vector of usage count filled with zeros.
    auto memrefTy = arg.getType().dyn_cast<hir::MemrefType>();
    if (!memrefTy)
      continue;
    auto numPorts = inputAttrs[i]
                        .getNamed("ports")
                        .getValue()
                        .second.dyn_cast<ArrayAttr>()
                        .size();
    SmallVector<SmallVector<Operation *>> perBankUses;
    perBankUses.append(memrefTy.getNumBanks(), SmallVector<Operation *>({}));
    mapMemref2PerPortPerBankUses[arg].append(numPorts, perBankUses);
  }
}

void MemrefFanoutInfo::visitOp(hir::AllocaOp op) {
  std::string moduleAttr =
      op.moduleAttr().dyn_cast<StringAttr>().getValue().str();

  auto numPorts = op.ports().size();
  // initialize the map with a vector of usage count filled with zeros.
  auto res = op.getResult();
  auto memrefTy = res.getType().dyn_cast<hir::MemrefType>();
  SmallVector<SmallVector<Operation *>> perBankUses;
  perBankUses.append(memrefTy.getNumBanks(), SmallVector<Operation *>({}));
  mapMemref2PerPortPerBankUses[res].append(numPorts, perBankUses);
}

void MemrefFanoutInfo::visitOp(hir::LoadOp op) {
  Value mem = op.mem();
  auto memrefTy = mem.getType().dyn_cast<hir::MemrefType>();
  assert(mapMemref2PerPortPerBankUses.find(mem) !=
         mapMemref2PerPortPerBankUses.end());
  int64_t linearIdx = helper::calcLinearIndex(op.filerIndices(BANK),
                                              memrefTy.filterShape(BANK));
  // before lowering explicit port must be specified.
  assert(op.port().hasValue());
  auto port = op.port().getValue();
  mapMemref2PerPortPerBankUses[mem][port][linearIdx].push_back(op);
}

void MemrefFanoutInfo::visitOp(hir::StoreOp op) {
  Value mem = op.mem();
  auto memrefTy = mem.getType().dyn_cast<hir::MemrefType>();
  assert(mapMemref2PerPortPerBankUses.find(mem) !=
         mapMemref2PerPortPerBankUses.end());
  int64_t linearIdx = helper::calcLinearIndex(op.filerIndices(BANK),
                                              memrefTy.filterShape(BANK));
  // before lowering explicit port must be specified.
  assert(op.port().hasValue());
  auto port = op.port().getValue();
  mapMemref2PerPortPerBankUses[mem][port][linearIdx].push_back(op);
}
