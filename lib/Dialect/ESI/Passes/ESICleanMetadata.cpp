//===- ESICleanMetadata.cpp - Clean ESI metadata ----------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// ESI clean metadata pass.
//===----------------------------------------------------------------------===//

#include "../PassDetails.h"

#include "circt/Dialect/ESI/ESIOps.h"
#include "circt/Dialect/ESI/ESIPasses.h"

namespace circt {
namespace esi {
#define GEN_PASS_DEF_ESICLEANMETADATA
#include "circt/Dialect/ESI/ESIPasses.h.inc"
} // namespace esi
} // namespace circt

using namespace circt;
using namespace esi;

namespace {
struct ESICleanMetadataPass
    : public circt::esi::impl::ESICleanMetadataBase<ESICleanMetadataPass> {
  void runOnOperation() override;
};
} // anonymous namespace

void ESICleanMetadataPass::runOnOperation() {
  auto mod = getOperation();

  mod.walk([&](Operation *op) {
    TypeSwitch<Operation *>(op)
        .Case<ServiceDeclOpInterface, ServiceImplRecordOp,
              ServiceRequestRecordOp, AppIDHierRootOp, IsManifestData>(
            [](Operation *op) { op->erase(); });
  });
}

std::unique_ptr<OperationPass<ModuleOp>>
circt::esi::createESICleanMetadataPass() {
  return std::make_unique<ESICleanMetadataPass>();
}
