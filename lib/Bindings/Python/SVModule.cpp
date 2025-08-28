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

#include "mlir/Bindings/Python/NanobindAdaptors.h"
#include "mlir/CAPI/IR.h"
#include "mlir/CAPI/Support.h"

#include "llvm/ADT/SmallVector.h"

#include "NanobindUtils.h"
#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>
namespace nb = nanobind;

using namespace mlir::python::nanobind_adaptors;

void circt::python::populateDialectSVSubmodule(nb::module_ &m) {
  m.doc() = "SV Python Native Extension";

  mlir_attribute_subclass(m, "SVAttributeAttr", svAttrIsASVAttributeAttr)
      .def_classmethod(
          "get",
          [](nb::object cls, std::string name, nb::object expressionObj,
             nb::object emitAsCommentObj, MlirContext ctxt) {
            // Set emitAsComment from optional boolean flag.
            bool emitAsComment = false;
            if (!emitAsCommentObj.is_none())
              emitAsComment = nb::cast<bool>(emitAsCommentObj);

            // Need temporary storage for casted string.
            std::string expr;
            MlirStringRef expression = {nullptr, 0};
            if (!expressionObj.is_none()) {
              expr = nb::cast<std::string>(expressionObj);
              expression = mlirStringRefCreateFromCString(expr.c_str());
            }
            return cls(svSVAttributeAttrGet(
                ctxt, mlirStringRefCreateFromCString(name.c_str()), expression,
                emitAsComment));
          },
          "Create a SystemVerilog attribute", nb::arg(), nb::arg("name"),
          nb::arg("expression") = nb::none(),
          nb::arg("emit_as_comment") = nb::none(), nb::arg("ctxt") = nb::none())
      .def_property_readonly("name",
                             [](MlirAttribute self) {
                               MlirStringRef name =
                                   svSVAttributeAttrGetName(self);
                               return std::string(name.data, name.length);
                             })
      .def_property_readonly("expression",
                             [](MlirAttribute self) -> nb::object {
                               MlirStringRef name =
                                   svSVAttributeAttrGetExpression(self);
                               if (name.data == nullptr)
                                 return nb::none();
                               return nb::str(name.data, name.length);
                             })
      .def_property_readonly("emit_as_comment", [](MlirAttribute self) {
        return svSVAttributeAttrGetEmitAsComment(self);
      });

  mlir_type_subclass(m, "InterfaceType", svTypeIsAInterfaceType)
      .def_classmethod(
          "get",
          [](nb::object cls, std::string interfaceSym, MlirContext ctxt) {
            return cls(svInterfaceTypeGet(
                ctxt, mlirStringRefCreateFromCString(interfaceSym.c_str())));
          },
          "Create a SystemVerilog InterfaceType", nb::arg(),
          nb::arg("interface_sym"), nb::arg("ctxt") = nb::none())
      .def_property_readonly("interface_sym", [](MlirType self) {
        return svInterfaceTypeGetInterfaceSym(self);
      });
}
