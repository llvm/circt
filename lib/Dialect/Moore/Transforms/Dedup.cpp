//===- Dedup.cpp - Moore module deduping --------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file  implements moore module deduplication.
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/Moore/MooreOps.h"
#include "circt/Dialect/Moore/MoorePasses.h"
#include "circt/Dialect/Moore/MooreTypes.h"
#include "circt/Support/LLVM.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LLVM.h"
#include "llvm/ADT/ScopedHashTable.h"

namespace circt {
namespace moore {
#define GEN_PASS_DEF_DEDUP
#include "circt/Dialect/Moore/MoorePasses.h.inc"
} // namespace moore
} // namespace circt

using namespace circt;
using namespace moore;

namespace {
class DedupPass : public circt::moore::impl::DedupBase<DedupPass> {
  // This table contains information to determine module module uniqueness.
  using StructuralHash = std::array<unsigned, 32>;
  using ModuleInfoTable = DenseMap<mlir::StringAttr, StructuralHash>;
  ModuleInfoTable moduleInfoTable;
  // This tale records old module name and equiplance module name to update
  using Symbol2Symbol = DenseMap<mlir::StringAttr, mlir::StringAttr>;
  Symbol2Symbol replaceTable;

  void setHashValue(StructuralHash &val, int index, mlir::StringAttr str);
  void runOnOperation() override;
};
} // namespace

std::unique_ptr<mlir::Pass> circt::moore::createDedupPass() {
  return std::make_unique<DedupPass>();
}

void DedupPass::setHashValue(StructuralHash &val, int index,
                             mlir::StringAttr str) {
  // We assume SHA256 is already a good hash and just truncate down to the
  // number of bytes we need for DenseMap.
  unsigned hash;
  std::memcpy(&hash, str.str().c_str(), sizeof(unsigned));
  val[index] = hash;
}

void DedupPass::runOnOperation() {
  // Do equiplance and record in replacTable
  // Dedup already exist module op
  getOperation()->walk([&](SVModuleOp moduleOp) {
    mlir::OpBuilder builder(&getContext());
    // If this Op has alreday in replace Table, erase it.
    if (replaceTable.lookup(moduleOp.getSymNameAttr())) {
      moduleOp->erase();
      return WalkResult::advance();
    }

    // Do hash calculation
    StructuralHash moduleInfo;
    auto moduleName = moduleOp.getSymNameAttr();

    auto moduleType = moduleOp.getModuleTypeAttrName();
    setHashValue(moduleInfo, 0, moduleType);

    // Compare and record to replacetable and erase this op if there is a
    // equiplance
    for (auto existModuleInfo : moduleInfoTable) {
      if (moduleInfo == existModuleInfo.getSecond()) {
        replaceTable.insert({moduleName, existModuleInfo.getFirst()});
        moduleOp.erase();
        return WalkResult::advance();
      }
    }
    moduleInfoTable.insert({moduleName, moduleInfo});
    return WalkResult::advance();
  });

  // updata instanceOp module name to the new name
  getOperation()->walk([&](InstanceOp InstanceOp) {
    mlir::OpBuilder builder(&getContext());
    if (auto oldName =
            replaceTable.lookup(InstanceOp.getModuleNameAttrName())) {
      InstanceOp.setModuleName(replaceTable[oldName]);
    }
    return WalkResult::advance();
  });
}
