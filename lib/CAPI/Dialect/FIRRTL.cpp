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

void firrtlSetErrorHandler(FirrtlContext ctx, FirrtlErrorHandler handler,
                           void *userData) {
  auto *ffiCtx = unwrap(ctx);

  ffiCtx->setErrorHandler([callback = handler,
                           userData](std::string_view message) {
    callback(firrtlCreateStringRef(message.data(), message.size()), userData);
  });
}

void firrtlVisitCircuit(FirrtlContext ctx, FirrtlStringRef name) {
  auto *ffiCtx = unwrap(ctx);

  ffiCtx->visitCircuit(unwrap(name));
}

void firrtlVisitModule(FirrtlContext ctx, FirrtlStringRef name) {
  auto *ffiCtx = unwrap(ctx);

  ffiCtx->visitModule(unwrap(name));
}

void firrtlVisitExtModule(FirrtlContext ctx, FirrtlStringRef name,
                          FirrtlStringRef defName) {
  auto *ffiCtx = unwrap(ctx);

  ffiCtx->visitExtModule(unwrap(name), unwrap(defName));
}

void firrtlVisitParameter(FirrtlContext ctx, FirrtlStringRef name,
                          const FirrtlParameter *param) {
  auto *ffiCtx = unwrap(ctx);

  ffiCtx->visitParameter(unwrap(name), *param);
}

void firrtlVisitPort(FirrtlContext ctx, FirrtlStringRef name,
                     FirrtlPortDirection direction, const FirrtlType *type) {
  auto *ffiCtx = unwrap(ctx);

  firrtl::Direction dir;
  if (direction == FIRRTL_PORT_DIRECTION_INPUT) {
    dir = firrtl::Direction::In;
  } else if (direction == FIRRTL_PORT_DIRECTION_OUTPUT) {
    dir = firrtl::Direction::Out;
  } else {
    ffiCtx->emitError("invalid port direction");
    return;
  }

  ffiCtx->visitPort(unwrap(name), dir, *type);
}

MLIR_CAPI_EXPORTED void firrtlVisitDeclaration(FirrtlContext ctx,
                                               const FirrtlDeclaration *decl) {
  auto *ffiCtx = unwrap(ctx);

  ffiCtx->visitDeclaration(*decl);
}

void firrtlVisitStatement(FirrtlContext ctx, const FirrtlStatement *stmt) {
  auto *ffiCtx = unwrap(ctx);

  ffiCtx->visitStatement(*stmt);
}

FirrtlStringRef firrtlExportFirrtl(FirrtlContext ctx) {
  auto *ffiCtx = unwrap(ctx);

  std::string output;
  llvm::raw_string_ostream stream{output};
  ffiCtx->exportFIRRTL(stream);

  auto len = output.size();
  auto *rawCStr = new char[len + 1];
  std::memcpy(rawCStr, output.c_str(), len);
  rawCStr[len] = '\0';

  return firrtlCreateStringRef(rawCStr, len);
}

void firrtlDestroyString(FirrtlContext ctx, FirrtlStringRef string) {
  (void)ctx;
  delete[] string.data;
}
