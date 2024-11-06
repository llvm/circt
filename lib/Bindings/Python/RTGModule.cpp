//===- RTGModule.cpp - RTG API pybind module ------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "circt/Bindings/Python/CIRCTModules.h"

#include "circt-c/Dialect/RTG.h"

#include "mlir-c/BuiltinAttributes.h"
#include "mlir/Bindings/Python/PybindAdaptors.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/Support/raw_ostream.h"

#include "PybindUtils.h"
#include "mlir-c/Support.h"
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
          "get", [](py::object cls, MlirContext ctxt, py::list argTypes) {
            std::vector<MlirType> types;
            for (auto type : argTypes)
              types.push_back(type.cast<MlirType>());
            return cls(rtgSequenceTypeGet(ctxt, types.size(), types.data()));
          });

  mlir_type_subclass(m, "ModeType", rtgTypeIsAMode)
      .def_classmethod("get", [](py::object cls, MlirContext ctxt) {
        return cls(rtgModeTypeGet(ctxt));
      });

  mlir_type_subclass(m, "ContextResourceType", rtgTypeIsAContextResource)
      .def_classmethod("get", [](py::object cls, MlirContext ctxt) {
        return cls(rtgContextResourceTypeGet(ctxt));
      });

  mlir_type_subclass(m, "SetType", rtgTypeIsASet)
      .def_classmethod(
          "get", [](py::object cls, MlirContext ctxt, MlirType elementType) {
            return cls(rtgSetTypeGet(ctxt, elementType));
          });

  mlir_type_subclass(m, "TargetType", rtgTypeIsATarget)
      .def_classmethod("get", [](py::object cls, MlirContext ctxt,
                                 py::list entryNames, py::list entryTypes) {
        std::vector<MlirAttribute> names;
        std::vector<MlirType> types;
        for (auto type : entryNames)
          names.push_back(type.cast<MlirAttribute>());
        for (auto type : entryTypes)
          types.push_back(type.cast<MlirType>());
        assert(names.size() == types.size());
        return cls(
            rtgTargetTypeGet(ctxt, types.size(), names.data(), types.data()));
      });
}
