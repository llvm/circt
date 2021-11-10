//===- HandshakeWrapESIPass.cpp - Implement Handshake wrap ESI Pass -------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "PassDetails.h"
#include "circt/Dialect/ESI/ESIDialect.h"
#include "circt/Dialect/HW/HWOps.h"
#include "circt/Support/LLVM.h"
#include "mlir/IR/Builders.h"

using namespace circt;

namespace {
struct HandshakeWrapESIPass
    : public handshake::HandshakeWrapESIBase<HandshakeWrapESIPass> {
  void runOnOperation() override;
};
} // namespace

void HandshakeWrapESIPass::runOnOperation() {
  ModuleOp mod = getOperation();
  StringRef topLevelName = topModule.getValue();
  hw::HWModuleOp topLevel = mod.lookupSymbol<hw::HWModuleOp>(topLevelName);
  if (!topLevel)
    return signalPassFailure();

  SmallVector<esi::ESIPortValidReadyMapping, 8> liPorts;
  esi::findValidReadySignals(topLevel, liPorts);
  if (liPorts.empty())
    return signalPassFailure();

  OpBuilder builder = OpBuilder::atBlockEnd(mod.getBody());
  if (!esi::buildESIWrapper(builder, topLevel, liPorts))
    return signalPassFailure();
}

std::unique_ptr<mlir::Pass> circt::handshake::createHandshakeWrapESIPass() {
  return std::make_unique<HandshakeWrapESIPass>();
}
