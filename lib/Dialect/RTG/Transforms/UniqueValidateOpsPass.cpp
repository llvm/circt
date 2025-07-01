//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/RTG/IR/RTGOps.h"
#include "circt/Dialect/RTG/Transforms/RTGPasses.h"
#include "circt/Support/Namespace.h"

namespace circt {
namespace rtg {
#define GEN_PASS_DEF_UNIQUEVALIDATEOPSPASS
#include "circt/Dialect/RTG/Transforms/RTGPasses.h.inc"
} // namespace rtg
} // namespace circt

using namespace mlir;
using namespace circt;
using namespace circt::rtg;

//===----------------------------------------------------------------------===//
// Unique Validate Ops Pass
//===----------------------------------------------------------------------===//

namespace {
struct UniqueValidateOpsPass
    : public rtg::impl::UniqueValidateOpsPassBase<UniqueValidateOpsPass> {
  using Base::Base;

  void runOnOperation() override;
};
} // namespace

void UniqueValidateOpsPass::runOnOperation() {
  auto moduleOp = getOperation();
  Namespace names;
  SmallVector<ValidateOp> validateOps;

  // Collect all the already fixed names in a first iteration.
  moduleOp.walk([&](ValidateOp op) {
    if (op.getId().has_value())
      names.add(op.getId().value());
    else
      validateOps.push_back(op);
  });

  for (auto op : validateOps) {
    auto newName = names.newName("validation_id");
    op.setId(newName);
  }
}
