//===- SeqModule.cpp - Seq API pybind module ------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "DialectModules.h"

#include "circt-c/Dialect/Seq.h"
#include "mlir/Bindings/Python/PybindAdaptors.h"

#include "PybindUtils.h"
#include "mlir-c/Support.h"
#include <pybind11/pybind11.h>
#include <pybind11/pytypes.h>
#include <pybind11/stl.h>

namespace py = pybind11;

using namespace circt;
using namespace mlir::python::adaptors;

/// Populate the seq python module.
void circt::python::populateDialectSeqSubmodule(py::module &m) {
  m.doc() = "Seq dialect Python native extension";

  mlir_type_subclass(m, "ClockType", seqTypeIsAClock)
      .def_classmethod(
          "get",
          [](py::object cls, MlirContext ctx) {
            return cls(seqClockTypeGet(ctx));
          },
          py::arg("cls"), py::arg("context") = py::none());
}
