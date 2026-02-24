//===- SVModule.cpp - SV API nanobind module ------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "CIRCTModules.h"

#include "circt-c/Dialect/SV.h"
#include "mlir-c/Bindings/Python/Interop.h"

#include "mlir/Bindings/Python/IRCore.h"
#include "mlir/CAPI/IR.h"
#include "mlir/CAPI/Support.h"

#include "llvm/ADT/SmallVector.h"

#include "NanobindUtils.h"
#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>
namespace nb = nanobind;

using mlir::python::MLIR_BINDINGS_PYTHON_DOMAIN::DefaultingPyMlirContext;
using mlir::python::MLIR_BINDINGS_PYTHON_DOMAIN::PyConcreteAttribute;
using mlir::python::MLIR_BINDINGS_PYTHON_DOMAIN::PyMlirContext;

struct PySVAttributeAttr : PyConcreteAttribute<PySVAttributeAttr> {
  static constexpr IsAFunctionTy isaFunction = svAttrIsASVAttributeAttr;
  static constexpr const char *pyClassName = "SVAttributeAttr";
  using Base::Base;

  static void bindDerived(ClassTy &c) {
    c.def_static(
        "get",
        [](std::string name, nb::object expressionObj,
           nb::object emitAsCommentObj, MlirContext ctxt) {
          bool emitAsComment = false;
          if (!emitAsCommentObj.is_none())
            emitAsComment = nb::cast<bool>(emitAsCommentObj);

          std::string expr;
          MlirStringRef expression = {nullptr, 0};
          if (!expressionObj.is_none()) {
            expr = nb::cast<std::string>(expressionObj);
            expression = mlirStringRefCreateFromCString(expr.c_str());
          }
          auto attr = svSVAttributeAttrGet(
              ctxt, mlirStringRefCreateFromCString(name.c_str()), expression,
              emitAsComment);
          return PySVAttributeAttr(
              PyMlirContext::forContext(mlirAttributeGetContext(attr)), attr);
        },
        "Create a SystemVerilog attribute", nb::arg("name"),
        nb::arg("expression") = nb::none(),
        nb::arg("emit_as_comment") = nb::none(),
        nb::arg("ctxt") = nb::none());
    c.def_prop_ro("name",
                  [](PySVAttributeAttr &self) {
                    MlirStringRef name = svSVAttributeAttrGetName(self);
                    return std::string(name.data, name.length);
                  });
    c.def_prop_ro("expression",
                  [](PySVAttributeAttr &self) -> nb::object {
                    MlirStringRef name = svSVAttributeAttrGetExpression(self);
                    if (name.data == nullptr)
                      return nb::none();
                    return nb::str(name.data, name.length);
                  });
    c.def_prop_ro("emit_as_comment", [](PySVAttributeAttr &self) {
      return svSVAttributeAttrGetEmitAsComment(self);
    });
  }
};

void circt::python::populateDialectSVSubmodule(nb::module_ &m) {
  m.doc() = "SV Python Native Extension";

  PySVAttributeAttr::bind(m);
}
