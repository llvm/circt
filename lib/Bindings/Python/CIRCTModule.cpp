//===- CIRCTModule.cpp - Main pybind module -------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "circt-c/Dialect/Comb.h"
#include "circt-c/Dialect/RTL.h"
#include "circt-c/Dialect/SV.h"
#include "mlir-c/Bindings/Python/Interop.h"
#include "mlir-c/Registration.h"

#include <pybind11/pybind11.h>
namespace py = pybind11;

PYBIND11_MODULE(_circt, m) {
  m.doc() = "CIRCT Python Native Extension";

  m.def(
      "register_dialects",
      [](py::object capsule) {
        // Get the MlirContext capsule from PyMlirContext capsule.
        auto wrappedCapsule = capsule.attr(MLIR_PYTHON_CAPI_PTR_ATTR);
        MlirContext context = mlirPythonCapsuleToContext(wrappedCapsule.ptr());

        // Collect CIRCT dialects to register.
        MlirDialectHandle rtl = mlirGetDialectHandle__rtl__();
        mlirDialectHandleRegisterDialect(rtl, context);
        mlirDialectHandleLoadDialect(rtl, context);
        MlirDialectHandle comb = mlirGetDialectHandle__comb__();
        mlirDialectHandleRegisterDialect(comb, context);
        mlirDialectHandleLoadDialect(comb, context);
        MlirDialectHandle sv = mlirGetDialectHandle__sv__();
        mlirDialectHandleRegisterDialect(sv, context);
        mlirDialectHandleLoadDialect(sv, context);
      },
      "Register CIRCT dialects on a PyMlirContext.");
}
