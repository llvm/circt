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
#include "circt/Support/LLVM.h"
#include "mlir-c/Bindings/Python/Interop.h"

#include "mlir/CAPI/IR.h"
#include "mlir/CAPI/Support.h"
#include "mlir/IR/Builders.h"
#include "mlir/Parser.h"
#include "mlir/Support/FileUtilities.h"

#include "llvm/ADT/SmallVector.h"

#include "PybindUtils.h"
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
namespace py = pybind11;

using namespace circt;
using namespace circt::esi;

//===----------------------------------------------------------------------===//
// Functions that translate from something Pybind11 understands to MLIR C++.
//===----------------------------------------------------------------------===//

static MlirOperation pyWrapModule(MlirOperation cModOp,
                                  std::vector<std::string> portNames) {
  mlir::Operation *modOp = unwrap(cModOp);
  SmallVector<StringRef, 8> portNamesRefs;
  for (auto name : portNames)
    portNamesRefs.push_back(name);
  SmallVector<ESIPortValidReadyMapping, 8> portTriples;
  resolvePortNames(modOp, portNamesRefs, portTriples);
  OpBuilder b(modOp);
  Operation *wrapper = buildESIWrapper(b, modOp, portTriples);
  return wrap(wrapper);
}

//===----------------------------------------------------------------------===//
// The main entry point into the ESI API.
//===----------------------------------------------------------------------===//

class System {
public:
  System(MlirContext cCtxt) : ctxt(unwrap(cCtxt)) {
    module = OwningModuleRef(ModuleOp::create(UnknownLoc::get(ctxt)));
  }

  void loadMlir(std::string filename) {
    auto loadedMod = mlir::parseSourceFile(filename, ctxt);
    Block *loadedBlock = loadedMod->getBody();
    auto &ops = module->getBody()->getOperations();
    ops.splice(ops.end(), loadedBlock->getOperations());
  }

  MlirOperation get() { return wrap((Operation *)module.get()); }
  MlirOperation lookup(std::string symbol) {
    Operation *found =
        SymbolTable::lookupSymbolIn((Operation *)module.get(), symbol);
    return wrap(found);
  }

private:
  MLIRContext *ctxt;
  OwningModuleRef module;
};

void circt::python::populateDialectESISubmodule(py::module &m) {
  m.doc() = "ESI Python Native Extension";

  m.def("buildWrapper", &pyWrapModule,
        "Construct an ESI wrapper around RTL module 'op' given a list of "
        "latency-insensitive ports.",
        py::arg("op"), py::arg("name_list"));

  py::class_<System>(m, "System")
      .def(py::init<MlirContext>())
      .def("load_mlir", &System::loadMlir, "Load an MLIR assembly file.")
      .def("get", &System::get, "Get the top level module op.")
      .def("lookup", &System::lookup, "Lookup an RTL module and return it.");
}
