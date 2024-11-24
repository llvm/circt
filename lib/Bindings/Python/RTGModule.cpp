//===- RTGModule.cpp - RTG API pybind module ------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "CIRCTModules.h"

#include "circt-c/Dialect/RTG.h"

#include "mlir/Bindings/Python/PybindAdaptors.h"

#include <pybind11/pybind11.h>
#include <pybind11/pytypes.h>
#include <pybind11/stl.h>
namespace py = pybind11;

using namespace circt;
using namespace mlir::python::adaptors;

/// Populate the rtg python module.
void circt::python::populateDialectRTGSubmodule(py::module &m) {
  m.doc() = "RTG dialect Python native extension";

  mlir_type_subclass(m, "SequenceType", rtgTypeIsASequence)
      .def_classmethod(
          "get",
          [](py::object cls, MlirContext ctxt) {
            return cls(rtgSequenceTypeGet(ctxt));
          },
          py::arg("self"), py::arg("ctxt") = nullptr);

  mlir_type_subclass(m, "SetType", rtgTypeIsASet)
      .def_classmethod(
          "get",
          [](py::object cls, MlirType elementType) {
            return cls(rtgSetTypeGet(elementType));
          },
          py::arg("self"), py::arg("element_type"));

  mlir_type_subclass(m, "DictType", rtgTypeIsADict)
      .def_classmethod(
          "get",
          [](py::object cls, MlirContext ctxt, py::list entryNames,
             py::list entryTypes) {
            std::vector<MlirAttribute> names;
            std::vector<MlirType> types;
            for (auto type : entryNames)
              names.push_back(type.cast<MlirAttribute>());
            for (auto type : entryTypes)
              types.push_back(type.cast<MlirType>());
            assert(names.size() == types.size() &&
                   "number of entry names and entry types must match");
            return cls(
                rtgDictTypeGet(ctxt, types.size(), names.data(), types.data()));
          },
          py::arg("self"), py::arg("ctxt") = nullptr,
          py::arg("entry_names") = py::list(),
          py::arg("entry_types") = py::list());
}
