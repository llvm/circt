//===- CheckMultipleClockSources.cpp - Diagnose multi-clock results -------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/Seq/SeqPasses.h"
#include "mlir/IR/BuiltinOps.h"
#include "llvm/ADT/STLExtras.h"

namespace circt {
namespace seq {
#define GEN_PASS_DEF_CHECKMULTIPLECLOCKSOURCES
#include "circt/Dialect/Seq/SeqPasses.h.inc"
} // namespace seq
} // namespace circt

using namespace circt;
using namespace seq;
using namespace mlir;

namespace {

static constexpr StringLiteral clockDomainsAttrName = "seq.clock_domains";

struct CheckMultipleClockSourcesPass
    : public seq::impl::CheckMultipleClockSourcesBase<
          CheckMultipleClockSourcesPass> {
  using CheckMultipleClockSourcesBase::CheckMultipleClockSourcesBase;

  void runOnOperation() override {
    getOperation()->walk([&](Operation *operation) {
      auto domainSets =
          dyn_cast_or_null<ArrayAttr>(operation->getAttr(clockDomainsAttrName));
      if (!domainSets || domainSets.size() != operation->getNumResults())
        return;

      for (auto [index, domainSet] : llvm::enumerate(domainSets)) {
        auto domains = dyn_cast<ArrayAttr>(domainSet);
        if (!domains || domains.size() <= 1)
          continue;

        auto diagnostic = operation->emitRemark()
                          << "result #" << index
                          << " has multiple possible clock sources: ";
        llvm::interleaveComma(domains, diagnostic, [&](Attribute domain) {
          if (auto string = dyn_cast<StringAttr>(domain))
            diagnostic << string.getValue();
          else
            diagnostic << domain;
        });
      }
    });
  }
};

} // namespace
