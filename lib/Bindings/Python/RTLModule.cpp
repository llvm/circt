//===- RTLModule.cpp - RTL API pybind module ------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "DialectModules.h"

#include "circt-c/Dialect/RTL.h"

#include "MLIRPybindAdaptors.h"
#include "PybindUtils.h"
#include "llvm/Support/raw_ostream.h"
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
namespace py = pybind11;

using namespace circt;
using namespace mlir::python::adaptors;

/// Populate the rtl python module.
void circt::python::populateDialectRTLSubmodule(py::module &m) {
  m.doc() = "RTL dialect Python native extension";

  mlir_type_subclass(m, "ArrayType", rtlTypeIsAArrayType)
      .def_static("get",
                  [](MlirType elementType, intptr_t size) {
                    return py::cast(rtlArrayTypeGet(elementType, size));
                  })
      .def_property_readonly(
          "element_type",
          [](MlirType self) { return rtlArrayTypeGetElementType(self); })
      .def_property_readonly(
          "size", [](MlirType self) { return rtlArrayTypeGetSize(self); });

  mlir_type_subclass(m, "TypeAliasType", rtlTypeIsATypeAlias)
      .def_static("get", [](std::string name, MlirType inner) {
        return py::cast(rtlTypeAliasTypeGet(
            mlirIdentifierGet(mlirTypeGetContext(inner),
                              mlirStringRefCreate(name.data(), name.size())),
            inner));
      });
}
