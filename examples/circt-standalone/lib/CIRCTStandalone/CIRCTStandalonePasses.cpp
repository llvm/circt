//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "CIRCTStandalone/CIRCTStandalonePasses.h"
#include "circt/Dialect/HW/HWOps.h"
#include "mlir/IR/BuiltinOps.h"

namespace circt {
namespace standalone {
#define GEN_PASS_DEF_RENAMEHWMODULE
#include "CIRCTStandalone/CIRCTStandalonePasses.h.inc"

namespace {
struct RenameHWModule : public impl::RenameHWModuleBase<RenameHWModule> {
  using impl::RenameHWModuleBase<RenameHWModule>::RenameHWModuleBase;

  void runOnOperation() override {
    for (auto module : getOperation().getOps<hw::HWModuleOp>())
      if (module.getName() == "bar")
        module.setName("foo");
  }
};
} // namespace

} // namespace standalone
} // namespace circt
