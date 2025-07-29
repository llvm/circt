//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/Emit/EmitOps.h"
#include "circt/Dialect/RTG/IR/RTGOps.h"
#include "circt/Dialect/RTG/Transforms/RTGPasses.h"
#include "circt/Support/Path.h"
#include "llvm/ADT/SmallString.h"

namespace circt {
namespace rtg {
#define GEN_PASS_DEF_INSERTTESTTOFILEMAPPINGPASS
#include "circt/Dialect/RTG/Transforms/RTGPasses.h.inc"
} // namespace rtg
} // namespace circt

using namespace mlir;
using namespace circt;
using namespace circt::rtg;

//===----------------------------------------------------------------------===//
// Insert Test To File Mapping Pass
//===----------------------------------------------------------------------===//

namespace {
struct InsertTestToFileMappingPass
    : public rtg::impl::InsertTestToFileMappingPassBase<
          InsertTestToFileMappingPass> {
  using Base::Base;
  void runOnOperation() override;
};
} // namespace

void InsertTestToFileMappingPass::runOnOperation() {
  SmallVector<TestOp> tests(getOperation().getOps<TestOp>());
  auto loc = getOperation().getLoc();
  if (!splitOutput) {
    OpBuilder builder = OpBuilder::atBlockEnd(getOperation().getBody());
    auto fileOp = emit::FileOp::create(builder, loc, path);
    builder.setInsertionPointToStart(fileOp.getBody());
    for (auto testOp : tests)
      emit::RefOp::create(builder, loc, testOp.getSymNameAttr());

    return;
  }

  if (path.empty() || path == "-") {
    emitError(loc, "path must be specified when split-output is set");
    return signalPassFailure();
  }

  for (auto testOp : tests) {
    OpBuilder builder = OpBuilder::atBlockEnd(getOperation().getBody());
    llvm::SmallString<128> filename(path.getValue());
    appendPossiblyAbsolutePath(filename, testOp.getSymName() + ".s");
    auto fileOp = emit::FileOp::create(builder, loc, filename);
    OpBuilder::InsertionGuard guard(builder);
    builder.setInsertionPointToStart(fileOp.getBody());
    emit::RefOp::create(builder, loc, testOp.getSymNameAttr());
  }
}
