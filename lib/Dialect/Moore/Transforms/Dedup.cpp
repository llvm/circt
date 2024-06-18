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
  using ModuleInfo = struct ModuleStruct {
    SmallVector<Type> portTypes;
    SmallVector<Type> inputTypes;
    SmallVector<Type> outputTypes;
  };
  friend bool operator==(const ModuleInfo &lhs, const ModuleInfo &rhs) {
    return lhs.portTypes == rhs.portTypes && lhs.inputTypes == rhs.inputTypes &&
           lhs.outputTypes == rhs.outputTypes;
  }

  using ModuleInfoTable = DenseMap<mlir::StringAttr, ModuleInfo>;
  ModuleInfoTable moduleInfoTable;

  // This tale records old module name and equiplance module name to update
  using Symbol2Symbol = DenseMap<mlir::StringAttr, mlir::StringAttr>;
  Symbol2Symbol replaceTable;

  void runOnOperation() override;
};
} // namespace

std::unique_ptr<mlir::Pass> circt::moore::createDedupPass() {
  return std::make_unique<DedupPass>();
}

void DedupPass::runOnOperation() {
  // Do equiplance and record in replacTable
  // Dedup already exist module op
  getOperation()->walk([&](SVModuleOp moduleOp) {
    // Define uniqueness
    auto moduleName = moduleOp.getSymNameAttr();
    auto moduleType = moduleOp.getModuleType();
    ModuleInfo moduleInfo = {moduleType.getPortTypes(),
                             moduleType.getInputTypes(),
                             moduleType.getOutputTypes()};

    // Compare and record to replacetable
    // erase this op if there is a equiplance
    for (auto existModuleInfo : moduleInfoTable) {
      if (existModuleInfo.second == moduleInfo) {
        moduleOp->erase();
        replaceTable.insert({moduleName, existModuleInfo.first});
        return WalkResult::advance();
      }
    }
    moduleInfoTable[moduleName] = moduleInfo;
    return WalkResult::advance();
  });

  // Referring to replacetable, replace instance's module name
  getOperation()->walk([&](InstanceOp instanceOp) {
    auto instanceName = instanceOp.getModuleNameAttr().getAttr();
    if (replaceTable.lookup(instanceName)) {
      instanceOp.setModuleName(replaceTable[instanceName]);
    }
    return WalkResult::advance();
  });
}
