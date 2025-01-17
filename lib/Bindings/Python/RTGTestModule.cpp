//===- RTGTestModule.cpp - RTGTest API pybind module ----------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "CIRCTModules.h"

#include "circt-c/Dialect/RTGTest.h"

#include "mlir/Bindings/Python/PybindAdaptors.h"

#include <pybind11/pybind11.h>
#include <pybind11/pytypes.h>
#include <pybind11/stl.h>
namespace py = pybind11;

using namespace circt;
using namespace mlir::python::adaptors;

/// Populate the rtgtest python module.
void circt::python::populateDialectRTGTestSubmodule(py::module &m) {
  m.doc() = "RTGTest dialect Python native extension";

  mlir_type_subclass(m, "CPUType", rtgtestTypeIsACPU)
      .def_classmethod(
          "get",
          [](py::object cls, MlirContext ctxt) {
            return cls(rtgtestCPUTypeGet(ctxt));
          },
          py::arg("self"), py::arg("ctxt") = nullptr);

  mlir_attribute_subclass(m, "CPUAttr", rtgtestAttrIsACPU)
      .def_classmethod(
          "get",
          [](py::object cls, unsigned id, MlirContext ctxt) {
            return cls(rtgtestCPUAttrGet(ctxt, id));
          },
          py::arg("self"), py::arg("id"), py::arg("ctxt") = nullptr)
      .def_property_readonly(
          "id", [](MlirAttribute self) { return rtgtestCPUAttrGetId(self); });
}
