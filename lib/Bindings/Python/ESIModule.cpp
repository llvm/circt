//===- ESIModule.cpp - ESI API pybind module ------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "DialectModules.h"

#include "circt-c/Dialect/ESI.h"
#include "circt/Dialect/ESI/ESIDialect.h"
#include "mlir-c/Bindings/Python/Interop.h"

#include <pybind11/pybind11.h>
namespace py = pybind11;

using namespace circt::esi;

//===----------------------------------------------------------------------===//
// Functions which should be part of the C API, but which are too difficult be
// bind to C. E.g. Functions which need complex data structures (like
// ArrayRefs). Pybind11 has conversions to/from the STL so it's easier to write
// a C++ wrapper.)
//===----------------------------------------------------------------------===//
// static pyFindValidReadySignals()
static MlirOperation pyWrapModule(PyInt MlirOperation op,
                                  std::vector<ESIPortValidReadyMapping> ports) {
}

void circt::python::populateDialectESISubmodule(py::module &m) {
  m.doc() = "ESI Python Native Extension";

  m.def("buildWrapper", &pyWrapModule);
}
