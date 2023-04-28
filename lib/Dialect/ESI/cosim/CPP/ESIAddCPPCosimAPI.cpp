//===- ESIAddCPPCosimAPI.cpp - ESI C++ cosim API addition -------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Add the C++ ESI Capnp cosim API into the current module.
//
//===----------------------------------------------------------------------===//

#include "../../PassDetails.h"

#include "circt/Dialect/ESI/ESIOps.h"
#include "circt/Dialect/ESI/ESIPasses.h"
#include "circt/Dialect/ESI/ESIServices.h"
#include "circt/Dialect/ESI/ESITypes.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/IndentedOstream.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/JSON.h"

#include "CPPCosimAPI.h"

#include <iostream>
#include <memory>

using namespace mlir;
using namespace circt;
using namespace circt::esi;
using namespace cppcosimapi;

namespace {
struct ESIAddCPPCosimAPIPass
    : public ESIAddCPPCosimAPIBase<ESIAddCPPCosimAPIPass> {
  void runOnOperation() override;

  LogicalResult emitAPI(llvm::raw_ostream &os);
};
} // anonymous namespace

LogicalResult ESIAddCPPCosimAPIPass::emitAPI(llvm::raw_ostream &os) {
  ModuleOp mod = getOperation();
  return exportCosimCPPAPI(mod, os);
}

void ESIAddCPPCosimAPIPass::runOnOperation() {
  ModuleOp mod = getOperation();
  auto *ctxt = &getContext();

  // Check for cosim endpoints in the design. If the design doesn't have any
  // we don't need a schema.
  WalkResult cosimWalk =
      mod.walk([](CosimEndpointOp _) { return WalkResult::interrupt(); });
  if (!cosimWalk.wasInterrupted())
    return;

  // Generate the API.
  std::string apiStrBuffer;
  llvm::raw_string_ostream os(apiStrBuffer);
  if (failed(emitAPI(os))) {
    mod.emitError("Failed to emit ESI C++ cosim API");
    return;
  }

  // And stuff if in a verbatim op with a filename, optionally.
  OpBuilder b = OpBuilder::atBlockEnd(mod.getBody());
  auto verbatim = b.create<sv::VerbatimOp>(b.getUnknownLoc(),
                                           StringAttr::get(ctxt, os.str()));
  if (!outputFile.empty()) {
    auto outputFileAttr =
        circt::hw::OutputFileAttr::getFromFilename(ctxt, outputFile);
    verbatim->setAttr("output_file", outputFileAttr);
  }

  if (toStdErr) {
    // Also Print the API to stderr.
    std::cerr << os.str();
  }
}

std::unique_ptr<OperationPass<ModuleOp>>
circt::esi::createESIAddCPPCosimAPIPass() {
  return std::make_unique<ESIAddCPPCosimAPIPass>();
}
