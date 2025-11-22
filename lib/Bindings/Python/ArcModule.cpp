//===- ArcModule.cpp - Arc API nanobind module --------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "CIRCTModules.h"

#include "circt-c/Dialect/Arc.h"
#include "mlir-c/IR.h"
#include "mlir/Bindings/Python/NanobindAdaptors.h"

#include <nanobind/nanobind.h>
namespace nb = nanobind;

using namespace mlir::python::nanobind_adaptors;

void circt::python::populateDialectArcSubmodule(nb::module_ &m) {
  m.doc() = "Arc dialect Python native extension";

  mlir_type_subclass(m, "StateType", arcTypeIsAState)
      .def_classmethod(
          "get",
          [](nb::object cls, MlirType innerType) {
            return cls(arcStateTypeGet(innerType));
          },
          nb::arg("cls"), nb::arg("inner_type"))
      .def_property_readonly(
          "type", [](MlirType self) { return arcStateTypeGetType(self); });

  mlir_type_subclass(m, "MemoryType", arcTypeIsAMemory)
      .def_classmethod(
          "get",
          [](nb::object cls, unsigned numWords, MlirType wordType,
             MlirType addressType) {
            return cls(arcMemoryTypeGet(numWords, wordType, addressType));
          },
          nb::arg("cls"), nb::arg("num_words"), nb::arg("word_type"),
          nb::arg("address_type"));

  mlir_type_subclass(m, "StorageType", arcTypeIsAStorage)
      .def_classmethod(
          "get",
          [](nb::object cls, MlirContext ctx, nb::object size) {
            if (size.is_none())
              return cls(arcStorageTypeGet(ctx));
            return cls(
                arcStorageTypeGetWithSize(ctx, nb::cast<unsigned>(size)));
          },
          nb::arg("cls"), nb::arg("context") = nb::none(),
          nb::arg("size") = nb::none());

  mlir_type_subclass(m, "SimModelInstanceType", arcTypeIsASimModelInstance)
      .def_classmethod(
          "get",
          [](nb::object cls, MlirAttribute model) {
            return cls(arcSimModelInstanceTypeGet(model));
          },
          nb::arg("cls"), nb::arg("model"));
}
