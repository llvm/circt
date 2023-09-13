//===- PyCDEModule.cpp - PyCDE pybind module ------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir-c/Bindings/Python/Interop.h"

#include "mlir/Bindings/Python/PybindAdaptors.h"
#include "mlir/CAPI/IR.h"
#include "mlir/CAPI/Support.h"

#include <map>
#include <vector>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
namespace py = pybind11;

namespace {
class PyAppIDIndex {
public:
  PyAppIDIndex(MlirModule cmod) {}

  class ChildAppIDs {
    friend class PyAppIDIndex;

    using InstancePath = std::vector<MlirAttribute>;
    std::map<MlirAttribute, InstancePath> appIDPaths;
  };

  ChildAppIDs get(MlirOperation container);

private:
  std::map<MlirOperation, ChildAppIDs> containerAppIDs;
};
} // namespace

using namespace mlir::python::adaptors;
PYBIND11_MODULE(_pycde, m) {
  py::class_<PyAppIDIndex>(m, "AppIDIndex")
      .def(py::init<MlirModule>())
      .def("get", &PyAppIDIndex::get);
}
