//===- RTLModule.cpp - RTL API pybind module ------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "DialectModules.h"

#include "circt-c/Dialect/HW.h"

#include "mlir-c/BuiltinAttributes.h"
#include "mlir/Bindings/Python/PybindAdaptors.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/Support/raw_ostream.h"

#include "PybindUtils.h"
#include <pybind11/pybind11.h>
#include <pybind11/pytypes.h>
#include <pybind11/stl.h>
namespace py = pybind11;

using namespace circt;
using namespace mlir::python::adaptors;

/// Populate the rtl python module.
void circt::python::populateDialectRTLSubmodule(py::module &m) {
  m.doc() = "RTL dialect Python native extension";

  mlir_type_subclass(m, "ArrayType", rtlTypeIsAArrayType)
      .def_staticmethod("get", [](MlirType elementType, intptr_t size) {
        return py::cast(rtlArrayTypeGet(elementType, size));
      });

  mlir_type_subclass(m, "StructType", rtlTypeIsAStructType)
      .def_staticmethod("get", [](py::list pyFieldInfos) {
        llvm::SmallVector<RTLStructFieldInfo> mlirFieldInfos;
        MlirContext ctx;

        // Since we're just passing string refs to the type constructor, copy
        // them into a temporary vector to give them all new addresses.
        llvm::SmallVector<llvm::SmallString<8>> names;
        for (size_t i = 0, e = pyFieldInfos.size(); i < e; ++i) {
          auto tuple = pyFieldInfos[i].cast<py::tuple>();
          auto type = tuple[1].cast<MlirType>();
          ctx = mlirTypeGetContext(type);
          names.emplace_back(tuple[0].cast<std::string>());
          mlirFieldInfos.push_back(RTLStructFieldInfo{
              mlirStringRefCreate(names[i].data(), names[i].size()),
              mlirTypeAttrGet(type)});
        }
        return py::cast(rtlStructTypeGet(ctx, mlirFieldInfos.size(),
                                         mlirFieldInfos.data()));
      });
}
