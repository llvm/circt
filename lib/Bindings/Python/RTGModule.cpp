//===- RTGModule.cpp - RTG API nanobind module ----------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "CIRCTModules.h"

#include "circt-c/Dialect/RTG.h"

#include "mlir/Bindings/Python/NanobindAdaptors.h"

#include <nanobind/nanobind.h>

namespace nb = nanobind;

using namespace circt;
using namespace mlir::python::nanobind_adaptors;

/// Populate the rtg python module.
void circt::python::populateDialectRTGSubmodule(nb::module_ &m) {
  m.doc() = "RTG dialect Python native extension";

  mlir_type_subclass(m, "SequenceType", rtgTypeIsASequence)
      .def_classmethod(
          "get",
          [](nb::object cls, MlirContext ctxt) {
            return cls(rtgSequenceTypeGet(ctxt));
          },
          nb::arg("self"), nb::arg("ctxt") = nullptr);

  mlir_type_subclass(m, "LabelType", rtgTypeIsALabel)
      .def_classmethod(
          "get",
          [](nb::object cls, MlirContext ctxt) {
            return cls(rtgLabelTypeGet(ctxt));
          },
          nb::arg("self"), nb::arg("ctxt") = nullptr);

  mlir_type_subclass(m, "SetType", rtgTypeIsASet)
      .def_classmethod(
          "get",
          [](nb::object cls, MlirType elementType) {
            return cls(rtgSetTypeGet(elementType));
          },
          nb::arg("self"), nb::arg("element_type"));

  mlir_type_subclass(m, "BagType", rtgTypeIsABag)
      .def_classmethod(
          "get",
          [](nb::object cls, MlirType elementType) {
            return cls(rtgBagTypeGet(elementType));
          },
          nb::arg("self"), nb::arg("element_type"));

  mlir_type_subclass(m, "DictType", rtgTypeIsADict)
      .def_classmethod(
          "get",
          [](nb::object cls, MlirContext ctxt,
             const std::vector<std::pair<MlirAttribute, MlirType>> &entries) {
            std::vector<MlirAttribute> names;
            std::vector<MlirType> types;
            for (auto entry : entries) {
              names.push_back(entry.first);
              types.push_back(entry.second);
            }
            return cls(
                rtgDictTypeGet(ctxt, types.size(), names.data(), types.data()));
          },
          nb::arg("self"), nb::arg("ctxt") = nullptr,
          nb::arg("entries") =
              std::vector<std::pair<MlirAttribute, MlirType>>());
}
