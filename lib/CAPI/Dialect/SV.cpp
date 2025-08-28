//===- SV.cpp - C interface for the SV dialect ----------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "circt-c/Dialect/SV.h"
#include "circt/Dialect/SV/SVAttributes.h"
#include "circt/Dialect/SV/SVDialect.h"
#include "circt/Dialect/SV/SVPasses.h"
#include "circt/Dialect/SV/SVTypes.h"
#include "mlir/CAPI/IR.h"
#include "mlir/CAPI/Registration.h"
#include "mlir/CAPI/Support.h"

using namespace circt::sv;

void registerSVPasses() { registerPasses(); }
MLIR_DEFINE_CAPI_DIALECT_REGISTRATION(SystemVerilog, sv, SVDialect)

bool svAttrIsASVAttributeAttr(MlirAttribute cAttr) {
  return llvm::isa<SVAttributeAttr>(unwrap(cAttr));
}

MlirAttribute svSVAttributeAttrGet(MlirContext cCtxt, MlirStringRef cName,
                                   MlirStringRef cExpression,
                                   bool emitAsComment) {
  mlir::MLIRContext *ctxt = unwrap(cCtxt);
  mlir::StringAttr expr;
  if (cExpression.data != nullptr)
    expr = mlir::StringAttr::get(ctxt, unwrap(cExpression));
  return wrap(
      SVAttributeAttr::get(ctxt, mlir::StringAttr::get(ctxt, unwrap(cName)),
                           expr, mlir::BoolAttr::get(ctxt, emitAsComment)));
}

MlirStringRef svSVAttributeAttrGetName(MlirAttribute cAttr) {
  return wrap(llvm::cast<SVAttributeAttr>(unwrap(cAttr)).getName().getValue());
}

MlirStringRef svSVAttributeAttrGetExpression(MlirAttribute cAttr) {
  auto expr = llvm::cast<SVAttributeAttr>(unwrap(cAttr)).getExpression();
  if (expr)
    return wrap(expr.getValue());
  return {nullptr, 0};
}

bool svSVAttributeAttrGetEmitAsComment(MlirAttribute attribute) {
  return llvm::cast<SVAttributeAttr>(unwrap(attribute))
      .getEmitAsComment()
      .getValue();
}
//===----------------------------------------------------------------------===//
// Type API.
//===----------------------------------------------------------------------===//

bool svTypeIsAInterfaceType(MlirType cAttr) {
  return llvm::isa<InterfaceType>(unwrap(cAttr));
}

MlirType svInterfaceTypeGet(MlirContext cCtxt, MlirStringRef cInterfaceSym) {
  mlir::MLIRContext *ctxt = unwrap(cCtxt);

  auto interfaceSym = mlir::FlatSymbolRefAttr::get(
      mlir::StringAttr::get(ctxt, unwrap(cInterfaceSym)));

  return wrap(InterfaceType::get(ctxt, interfaceSym));
}

MlirStringRef svInterfaceTypeGetInterfaceSym(MlirType cType) {
  return wrap(
      llvm::cast<InterfaceType>(unwrap(cType)).getInterface().getValue());
}
