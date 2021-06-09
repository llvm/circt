//=========- HIROpExtraClassDecl.cpp - extraClassDeclarations for Ops -----===//
//
// This file implements the extraClassDeclarations for HIR ops.
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/HIR/HIR.h"
#include "circt/Dialect/HIR/HIRDialect.h"

using namespace mlir;
using namespace hir;

SmallVector<Value> LoadOp::getBankedIdx() {
  SmallVector<Value> bankIdx;
  operand_range addr = this->addr();
  MemrefType memTy = this->mem().getType().dyn_cast<hir::MemrefType>();
  auto bankedDims = memTy.getBankedDims();
  for (int i = (int)memTy.getShape().size() - 1; i >= 0; i--) {
    bool isBankedDim = false;
    for (auto dim : bankedDims) {
      if (i == dim.dyn_cast<IntegerAttr>().getInt())
        isBankedDim = true;
    }

    if (isBankedDim) {
      auto idx = addr[addr.size() - 1 - i];
      bankIdx.push_back(idx);
    }
  }
  return bankIdx;
}

SmallVector<Value> LoadOp::getPackedIdx() {
  SmallVector<Value> packIdx;
  operand_range addr = this->addr();
  MemrefType memTy = this->mem().getType().dyn_cast<hir::MemrefType>();
  auto packedDims = memTy.getPackedDims();
  for (int i = (int)memTy.getShape().size() - 1; i >= 0; i--) {
    bool isPackedDim = false;
    for (auto dim : packedDims) {
      if (i == dim)
        isPackedDim = true;
    }

    if (isPackedDim) {
      auto idx = addr[addr.size() - 1 - i];
      packIdx.push_back(idx);
    }
  }
  return packIdx;
}
