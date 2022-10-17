//===- SVDialect.cpp - C Interface for the SV Dialect -------------------===//
//
//  Implements a C Interface for the SV Dialect
//
//===----------------------------------------------------------------------===//

#include "circt-c/Dialect/SV.h"
#include "circt/Dialect/SV/SVAttributes.h"
#include "circt/Dialect/SV/SVDialect.h"
#include "circt/Dialect/SV/SVPasses.h"
#include "mlir/CAPI/IR.h"
#include "mlir/CAPI/Registration.h"
#include "mlir/CAPI/Support.h"

using namespace circt::sv;

void registerSVPasses() { registerPasses(); }
MLIR_DEFINE_CAPI_DIALECT_REGISTRATION(SystemVerilog, sv, SVDialect)

bool svAttrIsASVAttributeAttr(MlirAttribute cAttr) {
  return unwrap(cAttr).isa<SVAttributeAttr>();
}

MlirAttribute svSVAttributeAttrGet(MlirContext cCtxt, MlirStringRef cName,
                                   MlirStringRef cExpression) {
  mlir::MLIRContext *ctxt = unwrap(cCtxt);
  mlir::StringAttr expr;
  if (cExpression.data != nullptr)
    expr = mlir::StringAttr::get(ctxt, unwrap(cExpression));
  return wrap(SVAttributeAttr::get(
      ctxt, mlir::StringAttr::get(ctxt, unwrap(cName)), expr));
}

MlirStringRef svSVAttributeAttrGetName(MlirAttribute cAttr) {
  return wrap(unwrap(cAttr).cast<SVAttributeAttr>().getName().getValue());
}

MlirStringRef svSVAttributeAttrGetExpression(MlirAttribute cAttr) {
  auto expr = unwrap(cAttr).cast<SVAttributeAttr>().getExpression();
  if (expr)
    return wrap(expr.getValue());
  return {nullptr, 0};
}

bool svAttrIsASVAttributesAttr(MlirAttribute cAttr) {
  return unwrap(cAttr).isa<SVAttributesAttr>();
}

MlirAttribute svSVAttributesAttrGet(MlirContext cCtxt, MlirAttribute attributes,
                                    bool emitAsComments) {
  mlir::MLIRContext *ctxt = unwrap(cCtxt);
  return wrap(SVAttributesAttr::get(ctxt,
                                    unwrap(attributes).cast<mlir::ArrayAttr>(),
                                    mlir::BoolAttr::get(ctxt, emitAsComments)));
}

MlirAttribute svSVAttributesAttrGetAttributes(MlirAttribute attributes) {
  return wrap(unwrap(attributes).cast<SVAttributesAttr>().getAttributes());
}

bool svSVAttributesAttrGetEmitAsComments(MlirAttribute attributes) {
  return unwrap(attributes)
      .cast<SVAttributesAttr>()
      .getEmitAsComments()
      .getValue();
}
