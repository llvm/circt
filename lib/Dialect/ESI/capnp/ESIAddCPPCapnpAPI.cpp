//===- ESIAddCPPCapnpAPI.cpp - Cap'N'Proto CPP API emission -----*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Emit the C++ ESI Capnp cosim API into the current module.
//
//===----------------------------------------------------------------------===//

#include "../PassDetails.h"

#include "circt/Support/IndentingOStream.h"

#include "circt/Dialect/Comb/CombOps.h"
#include "circt/Dialect/ESI/ESIOps.h"
#include "circt/Dialect/ESI/ESIPasses.h"
#include "circt/Dialect/ESI/ESIServices.h"
#include "circt/Dialect/ESI/ESITypes.h"
#include "circt/Dialect/HW/HWAttributes.h"
#include "circt/Dialect/HW/HWOps.h"
#include "circt/Dialect/SV/SVOps.h"
#include "circt/Support/BackedgeBuilder.h"
#include "circt/Support/LLVM.h"
#include "circt/Support/SymCache.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"

#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/JSON.h"

#include <memory>

#ifdef CAPNP
#include "capnp/ESICapnp.h"
#endif

using namespace mlir;
using namespace circt;
using namespace support;
using namespace circt::comb;
using namespace circt::esi;
using namespace circt::hw;
using namespace circt::sv;

namespace {
struct ESIAddCPPCapnpAPIPass
    : public ESIAddCPPCapnpAPIBase<ESIAddCPPCapnpAPIPass> {
  void runOnOperation() override;

  LogicalResult emitAPI(llvm::raw_ostream &os);
};
} // anonymous namespace

LogicalResult ESIAddCPPCapnpAPIPass::emitAPI(llvm::raw_ostream &os) {
  ModuleOp mod = getOperation();
  return exportCosimCPPAPI(mod, os);
}

void ESIAddCPPCapnpAPIPass::runOnOperation() {
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
    mod.emitError("Failed to emit ESI Capnp API");
    return;
  }

  // And stuff if in a verbatim op with a filename, optionally.
  OpBuilder b = OpBuilder::atBlockEnd(mod.getBody());
  auto verbatim = b.create<sv::VerbatimOp>(b.getUnknownLoc(),
                                           StringAttr::get(ctxt, os.str()));
  if (!outputFile.empty()) {
    auto outputFileAttr = OutputFileAttr::getFromFilename(ctxt, outputFile);
    verbatim->setAttr("output_file", outputFileAttr);
  }
}

std::unique_ptr<OperationPass<ModuleOp>>
circt::esi::createESIAddCPPCapnpAPIPass() {
  return std::make_unique<ESIAddCPPCapnpAPIPass>();
}
