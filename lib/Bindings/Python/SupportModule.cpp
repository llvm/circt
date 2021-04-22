//===- SupportModule.cpp - Support API pybind module ----------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "SupportModule.h"

#include "circt/Support/BackedgeBuilder.h"
#include "circt/Support/LLVM.h"

#include "mlir-c/IR.h"
#include "mlir/CAPI/IR.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Dialect.h"

#include "PybindUtils.h"
#include "pybind11/pybind11.h"
namespace py = pybind11;

using namespace circt;

//===----------------------------------------------------------------------===//
// Classes that translate from something Pybind11 understands to MLIR C++.
//===----------------------------------------------------------------------===//

namespace {

class PyBackedge {
public:
  PyBackedge(Backedge backedge) : backedge(backedge) {}

  MlirValue value() { return wrap(backedge); }

  void setValue(MlirValue newValue) { backedge.setValue(unwrap(newValue)); }

private:
  Backedge backedge;
};

class PyBackedgeBuilder {
public:
  PyBackedgeBuilder(MlirLocation loc)
      : builder(OpBuilder(unwrap(mlirLocationGetContext(loc)))),
        backedgeBuilder(BackedgeBuilder(builder, unwrap(loc))) {}

  PyBackedge get(MlirType type) {
    return PyBackedge(backedgeBuilder.get(unwrap(type)));
  }

private:
  OpBuilder builder;
  BackedgeBuilder backedgeBuilder;
};

} // namespace

void circt::python::populateSupportSubmodule(py::module &m) {
  py::class_<PyBackedge>(m, "Backedge")
      .def(py::init<Backedge>())
      .def("set_value", &PyBackedge::setValue, "Set the backedge value.")
      .def_property_readonly("value", &PyBackedge::value,
                             "Get the value from the backedge.");
  py::class_<PyBackedgeBuilder>(m, "BackedgeBuilder")
      .def(py::init<MlirLocation>())
      .def("get", &PyBackedgeBuilder::get, "Get a backedge builder.");
}
