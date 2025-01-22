//===- RTGTestModule.cpp - RTGTest API nanobind module --------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "CIRCTModules.h"

#include "circt-c/Dialect/RTGTest.h"

#include "mlir/Bindings/Python/NanobindAdaptors.h"

#include <nanobind/nanobind.h>
namespace nb = nanobind;

using namespace circt;
using namespace mlir::python::nanobind_adaptors;

/// Populate the rtgtest python module.
void circt::python::populateDialectRTGTestSubmodule(nb::module_ &m) {
  m.doc() = "RTGTest dialect Python native extension";

  mlir_type_subclass(m, "CPUType", rtgtestTypeIsACPU)
      .def_classmethod(
          "get",
          [](nb::object cls, MlirContext ctxt) {
            return cls(rtgtestCPUTypeGet(ctxt));
          },
          nb::arg("self"), nb::arg("ctxt") = nullptr);

  mlir_attribute_subclass(m, "CPUAttr", rtgtestAttrIsACPU)
      .def_classmethod(
          "get",
          [](nb::object cls, unsigned id, MlirContext ctxt) {
            return cls(rtgtestCPUAttrGet(ctxt, id));
          },
          nb::arg("self"), nb::arg("id"), nb::arg("ctxt") = nullptr)
      .def_property_readonly(
          "id", [](MlirAttribute self) { return rtgtestCPUAttrGetId(self); });
}
