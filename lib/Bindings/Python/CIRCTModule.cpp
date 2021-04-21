//===- CIRCTModule.cpp - Main pybind module -------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "DialectModules.h"

#include "circt-c/Dialect/Comb.h"
#include "circt-c/Dialect/ESI.h"
#include "circt-c/Dialect/RTL.h"
#include "circt-c/Dialect/SV.h"
#include "circt-c/ExportVerilog.h"
#include "mlir-c/Bindings/Python/Interop.h"
#include "mlir-c/Registration.h"

#include "llvm-c/ErrorHandling.h"

#include "PybindUtils.h"
#include <pybind11/pybind11.h>
namespace py = pybind11;

static void registerPasses() { registerSVPasses(); }

PYBIND11_MODULE(_circt, m) {
  m.doc() = "CIRCT Python Native Extension";
  registerPasses();
  LLVMEnablePrettyStackTrace();

  m.def(
      "register_dialects",
      [](py::object capsule) {
        // Get the MlirContext capsule from PyMlirContext capsule.
        auto wrappedCapsule = capsule.attr(MLIR_PYTHON_CAPI_PTR_ATTR);
        MlirContext context = mlirPythonCapsuleToContext(wrappedCapsule.ptr());

        // Collect CIRCT dialects to register.
        MlirDialectHandle comb = mlirGetDialectHandle__comb__();
        mlirDialectHandleRegisterDialect(comb, context);
        mlirDialectHandleLoadDialect(comb, context);

        MlirDialectHandle esi = mlirGetDialectHandle__esi__();
        mlirDialectHandleRegisterDialect(esi, context);
        mlirDialectHandleLoadDialect(esi, context);

        MlirDialectHandle rtl = mlirGetDialectHandle__rtl__();
        mlirDialectHandleRegisterDialect(rtl, context);
        mlirDialectHandleLoadDialect(rtl, context);

        MlirDialectHandle sv = mlirGetDialectHandle__sv__();
        mlirDialectHandleRegisterDialect(sv, context);
        mlirDialectHandleLoadDialect(sv, context);
      },
      "Register CIRCT dialects on a PyMlirContext.");

  m.def("export_verilog", [](MlirModule mod, py::object fileObject) {
    circt::python::PyFileAccumulator accum(fileObject, false);
    py::gil_scoped_release();
    mlirExportVerilog(mod, accum.getCallback(), accum.getUserData());
  });

  py::module esi = m.def_submodule("_esi", "ESI API");
  circt::python::populateDialectESISubmodule(esi);
}
