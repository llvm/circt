//===- FIRRTL.cpp - C Interface for the FIRRTL Dialect --------------------===//
//
//===----------------------------------------------------------------------===//

#include "circt-c/Dialect/FIRRTL.h"
#include "circt/Dialect/FIRRTL/CHIRRTLDialect.h"
#include "circt/Dialect/FIRRTL/FIRRTLDialect.h"
#include "circt/Dialect/FIRRTL/FIRRTLFFIContext.h"
#include "mlir/CAPI/IR.h"
#include "mlir/CAPI/Registration.h"
#include "mlir/CAPI/Support.h"

using namespace circt;
using namespace firrtl;
using namespace chirrtl;

MLIR_DEFINE_CAPI_DIALECT_REGISTRATION(FIRRTL, firrtl,
                                      circt::firrtl::FIRRTLDialect)

DEFINE_C_API_PTR_METHODS(FirrtlContext, chirrtl::FFIContext);

FirrtlContext firrtlCreateContext(void) {
  return wrap(new chirrtl::FFIContext);
}

void firrtlDestroyContext(FirrtlContext ctx) { delete unwrap(ctx); }

void firrtlVisitCircuit(FirrtlContext ctx, MlirStringRef name) {
  auto *ffiCtx = unwrap(ctx);

  ffiCtx->visitCircuit(unwrap(name));
}

void firrtlVisitModule(FirrtlContext ctx, MlirStringRef name) {
  auto *ffiCtx = unwrap(ctx);

  ffiCtx->visitModule(unwrap(name));
}

void firrtlSetErrorHandler(FirrtlContext ctx, FirrtlErrorHandler handler,
                           void *userData) {
  auto *ffiCtx = unwrap(ctx);

  ffiCtx->setErrorHandler(
      [callback = handler, userData](std::string_view message) {
        callback(mlirStringRefCreate(message.data(), message.size()), userData);
      });
}

MlirStringRef firrtlExportFirrtl(FirrtlContext ctx) {
  auto *ffiCtx = unwrap(ctx);

  std::string output;
  llvm::raw_string_ostream stream{output};
  ffiCtx->exportFIRRTL(stream);

  auto len = output.size();
  auto *rawCStr = new char[len + 1];
  std::memcpy(rawCStr, output.c_str(), len);
  rawCStr[len] = '\0';

  return mlirStringRefCreate(rawCStr, len);
}

void firrtlDestroyString(FirrtlContext ctx, MlirStringRef string) {
  (void)ctx;
  delete[] string.data;
}
