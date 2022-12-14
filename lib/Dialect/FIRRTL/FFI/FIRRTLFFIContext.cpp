//===- FIRRTLFFIContext.cpp - .fir to FIRRTL dialect parser ---------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This implements FFI for CIRCT FIRRTL.
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/FIRRTL/FIRRTLFFIContext.h"
#include "circt/Dialect/FIRRTL/CHIRRTLDialect.h"
#include "circt/Dialect/FIRRTL/FIREmitter.h"
#include "circt/Dialect/FIRRTL/FIRRTLDialect.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Location.h"

using namespace circt;
using namespace firrtl;
using namespace chirrtl;

// This macro returns the underlying value of a `RequireAssigned`, which
// requires that the value has been set previously, otherwise it will emit an
// error and return in the current function.
#define RA_EXPECT(var, ra)                                                     \
  if (!(ra).underlying.has_value()) {                                          \
    this->emitError("expected `" #ra "` to be set");                           \
    return;                                                                    \
  }                                                                            \
  var = (ra).underlying.value(); // NOLINT(bugprone-macro-parentheses)

FFIContext::FFIContext() : mlirCtx{std::make_unique<MLIRContext>()} {
  mlirCtx->loadDialect<CHIRRTLDialect>();
  mlirCtx->loadDialect<FIRRTLDialect, hw::HWDialect>();

  module = std::make_unique<mlir::ModuleOp>(mlir::ModuleOp::create(mockLoc()));
  opBuilder = std::make_unique<mlir::OpBuilder>(module->getBodyRegion());
}

void FFIContext::setErrorHandler(
    std::function<void(std::string_view message)> handler) {
  errorHandler = std::move(handler);
}

void FFIContext::visitCircuit(StringRef name) {
  circuitOp = opBuilder->create<CircuitOp>(mockLoc(), stringRefToAttr(name));
  // TODO
  // auto circuitTarget = ("~" + name.getValue()).str();
}

void FFIContext::visitModule(StringRef name) {
  RA_EXPECT(auto circuitOp, this->circuitOp);
  //
  auto builder = circuitOp.getBodyBuilder();

  auto moduleOp = builder.create<FModuleOp>(
      mockLoc(), stringRefToAttr(name),
      ArrayRef<PortInfo>{} /* TODO: portList, annotations */);
  // auto moduleTarget = (circuitTarget + "|" + name.getValue()).str();
}

void FFIContext::exportFIRRTL(llvm::raw_ostream &os) const {
  // TODO: check states first, otherwise a sigsegv will probably happen.

  auto result = exportFIRFile(*module, os);
  if (result.failed()) {
    emitError("failed to export FIRRTL");
  }
}

Location FFIContext::mockLoc() const {
  // no location info available
  return mlir::UnknownLoc::get(mlirCtx.get());
}

StringAttr FFIContext::stringRefToAttr(StringRef stringRef) {
  return StringAttr::get(mlirCtx.get(), stringRef);
}

void FFIContext::emitError(std::string_view message) const {
  if (errorHandler) {
    errorHandler(message);
  }
}

#undef RA_EXPECT
