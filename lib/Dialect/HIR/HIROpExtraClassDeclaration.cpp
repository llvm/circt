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
  for (auto dim : bankedDims) {
    auto idx = addr[addr.size() - 1 - dim.dyn_cast<IntegerAttr>().getInt()];
    bankIdx.push_back(idx);
  }
  return bankIdx;
}

SmallVector<Value> LoadOp::getPackedIdx() {
  SmallVector<Value> packedIdx;
  operand_range addr = this->addr();
  MemrefType memTy = this->mem().getType().dyn_cast<hir::MemrefType>();
  auto packedDims = memTy.getPackedDims();
  for (auto dim : packedDims) {
    auto idx = addr[addr.size() - 1 - dim];
    packedIdx.push_back(idx);
  }
  return packedIdx;
}
