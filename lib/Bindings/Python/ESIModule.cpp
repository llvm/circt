//===- ESIModule.cpp - ESI API pybind module ------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "DialectModules.h"

#include "circt-c/Dialect/ESI.h"
#include "mlir-c/Bindings/Python/Interop.h"

#include "mlir/Bindings/Python/PybindAdaptors.h"
#include "mlir/CAPI/IR.h"
#include "mlir/CAPI/Support.h"

#include "llvm/ADT/SmallVector.h"

#include "PybindUtils.h"
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
namespace py = pybind11;

//===----------------------------------------------------------------------===//
// The main entry point into the ESI Assembly API.
//===----------------------------------------------------------------------===//

/// TODO: move this to only use C-APIs.
class System {
public:
  /// Construct an ESI system. The Python bindings really want to own the MLIR
  /// objects, so we create them in Python and pass them into the constructor.
  System(MlirModule modOp)
      : cCtxt(mlirModuleGetContext(modOp)), cModuleOp(modOp) {}

  /// Load the contents of an MLIR asm file into the system module.
  void loadMlir(std::string filename) {
    circtESIAppendMlirFile(cModuleOp,
                           mlirStringRefCreateFromCString(filename.c_str()));
  }

  MlirOperation lookup(std::string symbol) {
    return circtESILookup(cModuleOp,
                          mlirStringRefCreateFromCString(symbol.c_str()));
  }

  void printCapnpSchema(py::object fileObject) {
    circt::python::PyFileAccumulator accum(fileObject, false);
    py::gil_scoped_release();
    circtESIExportCosimSchema(cModuleOp, accum.getCallback(),
                              accum.getUserData());
  }

private:
  MlirContext cCtxt;
  MlirModule cModuleOp;
};

using namespace mlir::python::adaptors;

void circt::python::populateDialectESISubmodule(py::module &m) {
  m.doc() = "ESI Python Native Extension";
  ::registerESIPasses();

  m.def(
      "buildWrapper",
      [](MlirOperation cModOp, std::vector<std::string> cPortNames) {
        llvm::SmallVector<MlirStringRef, 8> portNames;
        for (auto portName : cPortNames)
          portNames.push_back({portName.c_str(), portName.length()});
        return circtESIWrapModule(cModOp, portNames.size(), portNames.data());
      },
      "Construct an ESI wrapper around HW module 'op' given a list of "
      "latency-insensitive ports.",
      py::arg("op"), py::arg("name_list"));

  py::class_<System>(m, "CppSystem")
      .def(py::init<MlirModule>())
      .def("load_mlir", &System::loadMlir, "Load an MLIR assembly file.")
      .def("lookup", &System::lookup, "Lookup an HW module and return it.")
      .def("print_cosim_schema", &System::printCapnpSchema,
           "Print the cosim RPC schema");

  mlir_type_subclass(m, "ChannelType", circtESITypeIsAChannelType)
      .def_classmethod("get",
                       [](py::object cls, MlirType inner) {
                         return cls(circtESIChannelTypeGet(inner));
                       })
      .def_property_readonly(
          "inner", [](MlirType self) { return circtESIChannelGetInner(self); });
}
