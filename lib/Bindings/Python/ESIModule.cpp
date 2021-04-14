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
#include "circt/Dialect/ESI/ESITypes.h"
#include "circt/Support/LLVM.h"
#include "mlir-c/Bindings/Python/Interop.h"
#include "mlir-c/Diagnostics.h"

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

// This function taken from npcomp.
// Register a diagnostic handler that will redirect output to `sys.stderr`
// instead of a C/C++-level file abstraction. This ensures, for example,
// that mlir diagnostics emitted are correctly routed in Jupyter notebooks.
static MlirDiagnosticHandlerID
registerPythonSysStderrDiagnosticHandler(MlirContext context) {
  auto diagnosticHandler = [](MlirDiagnostic diagnostic,
                              void *) -> MlirLogicalResult {
    std::stringstream ss;
    auto stringCallback = [](MlirStringRef s, void *stringCallbackUserData) {
      auto *ssp = static_cast<std::stringstream *>(stringCallbackUserData);
      ssp->write(s.data, s.length);
    };
    mlirDiagnosticPrint(diagnostic, stringCallback, static_cast<void *>(&ss));
    // Use pybind11's print:
    // https://pybind11.readthedocs.io/en/stable/advanced/pycpp/utilities.html
    using namespace pybind11::literals;
    py::print(ss.str(), "file"_a = py::module_::import("sys").attr("stderr"));
    return mlirLogicalResultSuccess();
  };
  MlirDiagnosticHandlerID id = mlirContextAttachDiagnosticHandler(
      context, diagnosticHandler, nullptr, [](void *) { return; });
  // Ignore the ID. We intend to keep this handler for the entire lifetime
  // of this context.
  return id;
}

static MlirType channelType(MlirType cElem) {
  Type elemTy = unwrap(cElem);
  auto chanTy = ChannelPort::get(elemTy.getContext(), elemTy);
  return wrap(chanTy);
}

//===----------------------------------------------------------------------===//
// The main entry point into the ESI Assembly API.
//===----------------------------------------------------------------------===//

class System {
public:
  System(MlirContext cCtxt) : ctxt(unwrap(cCtxt)) {
    registerPythonSysStderrDiagnosticHandler(cCtxt);
  }
  ~System() { mlirContextDetachDiagnosticHandler(wrap(ctxt), diagID); }

  MlirModule createModule() {
    assert(!module);
    auto loc = UnknownLoc::get(ctxt);
    module = ModuleOp::create(loc);
    return wrap(module);
  }

  void loadMlir(std::string filename) {
    assert(module && "Must call create_module first()!");
    auto loadedMod = mlir::parseSourceFile(filename, ctxt);
    Block *loadedBlock = loadedMod->getBody();
    assert(!module->getRegions().empty());
    if (module.body().empty()) {
      module.body().push_back(loadedBlock);
      return;
    }
    auto &ops = module.getBody()->getOperations();
    ops.splice(ops.end(), loadedBlock->getOperations());
  }

  MlirOperation lookup(std::string symbol) {
    Operation *found = SymbolTable::lookupSymbolIn(module, symbol);
    return wrap(found);
  }

private:
  MLIRContext *ctxt;
  ModuleOp module;
  MlirDiagnosticHandlerID diagID;
};

void circt::python::populateDialectESISubmodule(py::module &m) {
  m.doc() = "ESI Python Native Extension";
  ::registerESIPasses();

  m.def("buildWrapper", &pyWrapModule,
        "Construct an ESI wrapper around RTL module 'op' given a list of "
        "latency-insensitive ports.",
        py::arg("op"), py::arg("name_list"));
  m.def("channel_type", &channelType,
        "Create an ESI channel type which wraps the argument type");

  py::class_<System>(m, "CppSystem")
      .def(py::init<MlirContext>())
      .def("load_mlir", &System::loadMlir, "Load an MLIR assembly file.")
      .def("lookup", &System::lookup, "Lookup an RTL module and return it.")
      .def("create_module", &System::createModule, "");
}
