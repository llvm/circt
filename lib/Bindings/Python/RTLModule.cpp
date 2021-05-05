//===- RTLModule.cpp - RTL API pybind module ------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "DialectModules.h"

#include "circt-c/Dialect/RTL.h"

#include "mlir-c/BuiltinAttributes.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/Support/raw_ostream.h"

#include "MLIRPybindAdaptors.h"
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
      .def_static("get",
                  [](MlirType elementType, intptr_t size) {
                    return py::cast(rtlArrayTypeGet(elementType, size));
                  })
      .def_property_readonly(
          "element_type",
          [](MlirType self) { return rtlArrayTypeGetElementType(self); })
      .def_property_readonly(
          "size", [](MlirType self) { return rtlArrayTypeGetSize(self); });

  mlir_type_subclass(m, "StructType", rtlTypeIsAStructType)
      .def_static("get", [](py::list pyFieldInfos) {
        llvm::SmallVector<RTLStructFieldInfo> mlirFieldInfos;
        llvm::SmallVector<llvm::SmallString<8>> names;
        MlirContext ctx;
        size_t i = 0;
        for (auto &it : pyFieldInfos) {
          auto tuple = it.cast<py::tuple>();
          auto name = tuple[0].cast<std::string>();
          names.push_back(llvm::SmallString<8>(name));
          auto type = tuple[1].cast<MlirType>();
          ctx = mlirTypeGetContext(type);
          mlirFieldInfos.push_back(RTLStructFieldInfo{
              mlirStringRefCreate(names[i].data(), names[i].size()),
              mlirTypeAttrGet(type)});
          ++i;
        }
        return py::cast(rtlStructTypeGet(ctx, mlirFieldInfos.size(),
                                         mlirFieldInfos.data()));
      });
}
