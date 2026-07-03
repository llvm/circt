//===- ArcModule.cpp - Arc API nanobind module ----------------------------===//
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
          "type", [](MlirType self) { return arcStateTypeGetType(self); })
      .def_property_readonly(
          "bit_width",
          [](MlirType self) { return arcStateTypeGetBitWidth(self); })
      .def_property_readonly("byte_width", [](MlirType self) {
        return arcStateTypeGetByteWidth(self);
      });

  mlir_type_subclass(m, "MemoryType", arcTypeIsAMemory)
      .def_classmethod(
          "get",
          [](nb::object cls, unsigned numWords, MlirType wordType,
             MlirType addressType) {
            return cls(arcMemoryTypeGet(numWords, wordType, addressType));
          },
          nb::arg("cls"), nb::arg("num_words"), nb::arg("word_type"),
          nb::arg("address_type"))
      .def_property_readonly(
          "num_words",
          [](MlirType self) { return arcMemoryTypeGetNumWords(self); })
      .def_property_readonly(
          "word_type",
          [](MlirType self) { return arcMemoryTypeGetWordType(self); })
      .def_property_readonly(
          "address_type",
          [](MlirType self) { return arcMemoryTypeGetAddressType(self); })
      .def_property_readonly(
          "stride", [](MlirType self) { return arcMemoryTypeGetStride(self); });

  mlir_type_subclass(m, "StorageType", arcTypeIsAStorage)
      .def_classmethod(
          "get",
          [](nb::object cls, MlirContext ctx) {
            return cls(arcStorageTypeGet(ctx));
          },
          nb::arg("cls"), nb::arg("context") = nb::none());

  mlir_type_subclass(m, "SimModelInstanceType", arcTypeIsASimModelInstance)
      .def_classmethod(
          "get",
          [](nb::object cls, MlirAttribute model) {
            return cls(arcSimModelInstanceTypeGet(model));
          },
          nb::arg("cls"), nb::arg("model"))
      .def_property_readonly("model", [](MlirType self) {
        return arcSimModelInstanceTypeGetModel(self);
      });
}
