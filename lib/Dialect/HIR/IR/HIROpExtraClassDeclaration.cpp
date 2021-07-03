//=========- HIROpExtraClassDecl.cpp - extraClassDeclarations for Ops -----===//
//
// This file implements the extraClassDeclarations for HIR ops.
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/HIR/IR/HIR.h"
#include "circt/Dialect/HIR/IR/HIRDialect.h"

using namespace circt;
using namespace hir;

namespace {
SmallVector<Value> filterIndices(DimKind idxKind, OperandRange indices,
                                 ArrayRef<DimKind> dimKinds) {
  SmallVector<Value> addrIndices;
  for (size_t i = 0; i < indices.size(); i++) {
    if (dimKinds[i] == idxKind) {
      auto idx = indices[i];
      addrIndices.push_back(idx);
    }
  }
  return addrIndices;
}
} // namespace
SmallVector<Value> LoadOp::filerIndices(DimKind idxKind) {

  OperandRange indices = this->indices();
  auto dimKinds =
      this->mem().getType().dyn_cast<hir::MemrefType>().getDimKinds();
  return filterIndices(idxKind, indices, dimKinds);
}

SmallVector<Value> StoreOp::filerIndices(DimKind idxKind) {

  OperandRange indices = this->indices();
  auto dimKinds =
      this->mem().getType().dyn_cast<hir::MemrefType>().getDimKinds();
  return filterIndices(idxKind, indices, dimKinds);
}

SmallVector<Value, 4> hir::FuncOp::getOperands() {
  SmallVector<Value, 4> operands;

  auto &entryBlock = this->getFuncBody().front();
  for (Value arg :
       entryBlock.getArguments().slice(0, entryBlock.getNumArguments() - 1))
    operands.push_back(arg);
  return operands;
}

SmallVector<Value, 4> hir::CallOp::getOperands() {
  SmallVector<Value, 4> operands;
  for (Value arg : this->operands().slice(0, this->getNumOperands() - 1))
    operands.push_back(arg);
  return operands;
}
