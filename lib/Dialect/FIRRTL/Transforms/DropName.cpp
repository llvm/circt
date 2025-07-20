//===- DropName.cpp - Drop Names  -----------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines the DropName pass.
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/FIRRTL/FIRRTLOps.h"
#include "circt/Dialect/FIRRTL/Passes.h"
#include "circt/Support/Naming.h"
#include "mlir/Pass/Pass.h"

namespace circt {
namespace firrtl {
#define GEN_PASS_DEF_DROPNAME
#include "circt/Dialect/FIRRTL/Passes.h.inc"
} // namespace firrtl
} // namespace circt

using namespace circt;
using namespace firrtl;

namespace {
struct DropNamesPass : public circt::firrtl::impl::DropNameBase<DropNamesPass> {
  using Base::Base;

  enum ModAction { Drop, Keep, Demote };

  void runOnOperation() override {
    size_t namesDropped = 0;
    size_t namesChanged = 0;
    if (preserveMode == PreserveValues::None) {
      // Drop all names.
      dropNamesIf(namesChanged, namesDropped, [](FNamableOp op) {
        if (isUselessName(op.getName()))
          return ModAction::Drop;
        return ModAction::Demote;
      });
    } else if (preserveMode == PreserveValues::Strip) {
      // Strip all names.
      dropNamesIf(namesChanged, namesDropped,
                  [](FNamableOp op) { return ModAction::Drop; });
    } else if (preserveMode == PreserveValues::Named) {
      // Drop the name if it isn't considered meaningful.
      dropNamesIf(namesChanged, namesDropped, [](FNamableOp op) {
        auto name = op.getName();
        if (isUselessName(name))
          return ModAction::Drop;
        if (name.starts_with("_"))
          return ModAction::Demote;
        return ModAction::Keep;
      });
    } else if (preserveMode == PreserveValues::All) {
      // Drop the name if it isn't considered meaningful.
      dropNamesIf(namesChanged, namesDropped, [](FNamableOp op) {
        if (isUselessName(op.getName()))
          return ModAction::Demote;
        return ModAction::Keep;
      });
    }
    numNamesConverted += namesChanged;
    numNamesDropped += namesDropped;
  }

private:
  size_t dropNamesIf(size_t &namesChanged, size_t &namesDropped,
                     llvm::function_ref<ModAction(FNamableOp)> pred) {
    size_t changedNames = 0;
    auto emptyNameAttr = StringAttr::get(&getContext(), "");
    auto droppableNameAttr =
        NameKindEnumAttr::get(&getContext(), NameKindEnum::DroppableName);
    getOperation()->walk([&](FNamableOp op) {
      switch (pred(op)) {
      case ModAction::Drop:
        op.setNameAttr(emptyNameAttr);
        op.setNameKindAttr(droppableNameAttr);
        ++namesDropped;
        break;
      case ModAction::Demote:
        op.setNameKindAttr(droppableNameAttr);
        ++namesChanged;
        break;
      default:
        break;
      }
    });
    return changedNames;
  }
};

} // end anonymous namespace
