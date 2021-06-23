//=========- HIROpExtraClassDecl.cpp - extraClassDeclarations for Ops -----===//
//
// This file implements the extraClassDeclarations for HIR ops.
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/HIR/IR/HIR.h"
#include "circt/Dialect/HIR/IR/HIRDialect.h"

using namespace circt;
using namespace hir;

SmallVector<Value> LoadOp::getBankIdx() {
  SmallVector<Value> bankIdx;
  operand_range indices = this->indices();
  MemrefType memTy = this->mem().getType().dyn_cast<hir::MemrefType>();
  auto bankDims = memTy.getBankDims();
  for (int i = (int)memTy.getShape().size() - 1; i >= 0; i--) {
    bool isBankDim = false;
    for (auto dim : bankDims) {
      if (i == dim.dyn_cast<IntegerAttr>().getInt())
        isBankDim = true;
    }

    if (isBankDim) {
      auto idx = indices[indices.size() - 1 - i];
      bankIdx.push_back(idx);
    }
  }
  return bankIdx;
}

SmallVector<Value> LoadOp::getAddrIdx() {
  SmallVector<Value> addrIdx;
  operand_range indices = this->indices();
  MemrefType memTy = this->mem().getType().dyn_cast<hir::MemrefType>();
  auto addrDims = memTy.getAddrDims();
  for (int i = (int)memTy.getShape().size() - 1; i >= 0; i--) {
    bool isAddrDim = false;
    for (auto dim : addrDims) {
      if (i == dim)
        isAddrDim = true;
    }

    if (isAddrDim) {
      auto idx = indices[indices.size() - 1 - i];
      addrIdx.push_back(idx);
    }
  }
  return addrIdx;
}

SmallVector<Value> StoreOp::getBankIdx() {
  SmallVector<Value> bankIdx;
  operand_range indices = this->indices();
  MemrefType memTy = this->mem().getType().dyn_cast<hir::MemrefType>();
  auto bankDims = memTy.getBankDims();
  for (int i = (int)memTy.getShape().size() - 1; i >= 0; i--) {
    bool isBankDim = false;
    for (auto dim : bankDims) {
      if (i == dim.dyn_cast<IntegerAttr>().getInt())
        isBankDim = true;
    }

    if (isBankDim) {
      auto idx = indices[indices.size() - 1 - i];
      bankIdx.push_back(idx);
    }
  }
  return bankIdx;
}

SmallVector<Value> StoreOp::getAddrIdx() {
  SmallVector<Value> addrIdx;
  operand_range indices = this->indices();
  MemrefType memTy = this->mem().getType().dyn_cast<hir::MemrefType>();
  auto addrDims = memTy.getAddrDims();
  for (int i = (int)memTy.getShape().size() - 1; i >= 0; i--) {
    bool isAddrDim = false;
    for (auto dim : addrDims) {
      if (i == dim)
        isAddrDim = true;
    }

    if (isAddrDim) {
      auto idx = indices[indices.size() - 1 - i];
      addrIdx.push_back(idx);
    }
  }
  return addrIdx;
}

SmallVector<Value, 4> hir::FuncOp::getOperands() {
  SmallVector<Value, 4> operands;

  auto &entryBlock = this->getBody().front();
  for (Value arg :
       entryBlock.getArguments().slice(0, entryBlock.getNumArguments() - 1))
    operands.push_back(arg);
  return operands;
}

SmallVector<Value, 4> hir::CallOp::getOperands() {
  ;
  SmallVector<Value, 4> operands;
  for (Value arg : this->operands().slice(0, this->getNumOperands() - 1))
    operands.push_back(arg);
  return operands;
}
