//===- RTGTestModule.cpp - RTGTest API nanobind module --------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "CIRCTModules.h"

#include "circt-c/Dialect/RTGTest.h"

#include "mlir/Bindings/Python/NanobindAdaptors.h"

#include <nanobind/nanobind.h>
namespace nb = nanobind;

using namespace circt;
using namespace mlir::python::nanobind_adaptors;

/// Populate the rtgtest python module.
void circt::python::populateDialectRTGTestSubmodule(nb::module_ &m) {
  m.doc() = "RTGTest dialect Python native extension";

  mlir_type_subclass(m, "CPUType", rtgtestTypeIsACPU)
      .def_classmethod(
          "get",
          [](nb::object cls, MlirContext ctxt) {
            return cls(rtgtestCPUTypeGet(ctxt));
          },
          nb::arg("self"), nb::arg("ctxt") = nullptr);

  mlir_type_subclass(m, "IntegerRegisterType", rtgtestTypeIsAIntegerRegister)
      .def_classmethod(
          "get",
          [](nb::object cls, MlirContext ctxt) {
            return cls(rtgtestIntegerRegisterTypeGet(ctxt));
          },
          nb::arg("self"), nb::arg("ctxt") = nullptr);

  mlir_attribute_subclass(m, "CPUAttr", rtgtestAttrIsACPU)
      .def_classmethod(
          "get",
          [](nb::object cls, unsigned id, MlirContext ctxt) {
            return cls(rtgtestCPUAttrGet(ctxt, id));
          },
          nb::arg("self"), nb::arg("id"), nb::arg("ctxt") = nullptr)
      .def_property_readonly(
          "id", [](MlirAttribute self) { return rtgtestCPUAttrGetId(self); });

  mlir_attribute_subclass(m, "RegZeroAttr", rtgtestAttrIsARegZero)
      .def_classmethod(
          "get",
          [](nb::object cls, MlirContext ctxt) {
            return cls(rtgtestRegZeroAttrGet(ctxt));
          },
          nb::arg("self"), nb::arg("ctxt") = nullptr);

  mlir_attribute_subclass(m, "RegRaAttr", rtgtestAttrIsARegRa)
      .def_classmethod(
          "get",
          [](nb::object cls, MlirContext ctxt) {
            return cls(rtgtestRegRaAttrGet(ctxt));
          },
          nb::arg("self"), nb::arg("ctxt") = nullptr);

  mlir_attribute_subclass(m, "RegSpAttr", rtgtestAttrIsARegSp)
      .def_classmethod(
          "get",
          [](nb::object cls, MlirContext ctxt) {
            return cls(rtgtestRegSpAttrGet(ctxt));
          },
          nb::arg("self"), nb::arg("ctxt") = nullptr);

  mlir_attribute_subclass(m, "RegGpAttr", rtgtestAttrIsARegGp)
      .def_classmethod(
          "get",
          [](nb::object cls, MlirContext ctxt) {
            return cls(rtgtestRegGpAttrGet(ctxt));
          },
          nb::arg("self"), nb::arg("ctxt") = nullptr);

  mlir_attribute_subclass(m, "RegTpAttr", rtgtestAttrIsARegTp)
      .def_classmethod(
          "get",
          [](nb::object cls, MlirContext ctxt) {
            return cls(rtgtestRegTpAttrGet(ctxt));
          },
          nb::arg("self"), nb::arg("ctxt") = nullptr);

  mlir_attribute_subclass(m, "RegT0Attr", rtgtestAttrIsARegT0)
      .def_classmethod(
          "get",
          [](nb::object cls, MlirContext ctxt) {
            return cls(rtgtestRegT0AttrGet(ctxt));
          },
          nb::arg("self"), nb::arg("ctxt") = nullptr);

  mlir_attribute_subclass(m, "RegT1Attr", rtgtestAttrIsARegT1)
      .def_classmethod(
          "get",
          [](nb::object cls, MlirContext ctxt) {
            return cls(rtgtestRegT1AttrGet(ctxt));
          },
          nb::arg("self"), nb::arg("ctxt") = nullptr);

  mlir_attribute_subclass(m, "RegT2Attr", rtgtestAttrIsARegT2)
      .def_classmethod(
          "get",
          [](nb::object cls, MlirContext ctxt) {
            return cls(rtgtestRegT2AttrGet(ctxt));
          },
          nb::arg("self"), nb::arg("ctxt") = nullptr);

  mlir_attribute_subclass(m, "RegS0Attr", rtgtestAttrIsARegS0)
      .def_classmethod(
          "get",
          [](nb::object cls, MlirContext ctxt) {
            return cls(rtgtestRegS0AttrGet(ctxt));
          },
          nb::arg("self"), nb::arg("ctxt") = nullptr);

  mlir_attribute_subclass(m, "RegS1Attr", rtgtestAttrIsARegS1)
      .def_classmethod(
          "get",
          [](nb::object cls, MlirContext ctxt) {
            return cls(rtgtestRegS1AttrGet(ctxt));
          },
          nb::arg("self"), nb::arg("ctxt") = nullptr);

  mlir_attribute_subclass(m, "RegA0Attr", rtgtestAttrIsARegA0)
      .def_classmethod(
          "get",
          [](nb::object cls, MlirContext ctxt) {
            return cls(rtgtestRegA0AttrGet(ctxt));
          },
          nb::arg("self"), nb::arg("ctxt") = nullptr);

  mlir_attribute_subclass(m, "RegA1Attr", rtgtestAttrIsARegA1)
      .def_classmethod(
          "get",
          [](nb::object cls, MlirContext ctxt) {
            return cls(rtgtestRegA1AttrGet(ctxt));
          },
          nb::arg("self"), nb::arg("ctxt") = nullptr);

  mlir_attribute_subclass(m, "RegA2Attr", rtgtestAttrIsARegA2)
      .def_classmethod(
          "get",
          [](nb::object cls, MlirContext ctxt) {
            return cls(rtgtestRegA2AttrGet(ctxt));
          },
          nb::arg("self"), nb::arg("ctxt") = nullptr);

  mlir_attribute_subclass(m, "RegA3Attr", rtgtestAttrIsARegA3)
      .def_classmethod(
          "get",
          [](nb::object cls, MlirContext ctxt) {
            return cls(rtgtestRegA3AttrGet(ctxt));
          },
          nb::arg("self"), nb::arg("ctxt") = nullptr);

  mlir_attribute_subclass(m, "RegA4Attr", rtgtestAttrIsARegA4)
      .def_classmethod(
          "get",
          [](nb::object cls, MlirContext ctxt) {
            return cls(rtgtestRegA4AttrGet(ctxt));
          },
          nb::arg("self"), nb::arg("ctxt") = nullptr);

  mlir_attribute_subclass(m, "RegA5Attr", rtgtestAttrIsARegA5)
      .def_classmethod(
          "get",
          [](nb::object cls, MlirContext ctxt) {
            return cls(rtgtestRegA5AttrGet(ctxt));
          },
          nb::arg("self"), nb::arg("ctxt") = nullptr);

  mlir_attribute_subclass(m, "RegA6Attr", rtgtestAttrIsARegA6)
      .def_classmethod(
          "get",
          [](nb::object cls, MlirContext ctxt) {
            return cls(rtgtestRegA6AttrGet(ctxt));
          },
          nb::arg("self"), nb::arg("ctxt") = nullptr);

  mlir_attribute_subclass(m, "RegA7Attr", rtgtestAttrIsARegA7)
      .def_classmethod(
          "get",
          [](nb::object cls, MlirContext ctxt) {
            return cls(rtgtestRegA7AttrGet(ctxt));
          },
          nb::arg("self"), nb::arg("ctxt") = nullptr);

  mlir_attribute_subclass(m, "RegS2Attr", rtgtestAttrIsARegS2)
      .def_classmethod(
          "get",
          [](nb::object cls, MlirContext ctxt) {
            return cls(rtgtestRegS2AttrGet(ctxt));
          },
          nb::arg("self"), nb::arg("ctxt") = nullptr);

  mlir_attribute_subclass(m, "RegS3Attr", rtgtestAttrIsARegS3)
      .def_classmethod(
          "get",
          [](nb::object cls, MlirContext ctxt) {
            return cls(rtgtestRegS3AttrGet(ctxt));
          },
          nb::arg("self"), nb::arg("ctxt") = nullptr);

  mlir_attribute_subclass(m, "RegS4Attr", rtgtestAttrIsARegS4)
      .def_classmethod(
          "get",
          [](nb::object cls, MlirContext ctxt) {
            return cls(rtgtestRegS4AttrGet(ctxt));
          },
          nb::arg("self"), nb::arg("ctxt") = nullptr);

  mlir_attribute_subclass(m, "RegS5Attr", rtgtestAttrIsARegS5)
      .def_classmethod(
          "get",
          [](nb::object cls, MlirContext ctxt) {
            return cls(rtgtestRegS5AttrGet(ctxt));
          },
          nb::arg("self"), nb::arg("ctxt") = nullptr);

  mlir_attribute_subclass(m, "RegS6Attr", rtgtestAttrIsARegS6)
      .def_classmethod(
          "get",
          [](nb::object cls, MlirContext ctxt) {
            return cls(rtgtestRegS6AttrGet(ctxt));
          },
          nb::arg("self"), nb::arg("ctxt") = nullptr);

  mlir_attribute_subclass(m, "RegS7Attr", rtgtestAttrIsARegS7)
      .def_classmethod(
          "get",
          [](nb::object cls, MlirContext ctxt) {
            return cls(rtgtestRegS7AttrGet(ctxt));
          },
          nb::arg("self"), nb::arg("ctxt") = nullptr);

  mlir_attribute_subclass(m, "RegS8Attr", rtgtestAttrIsARegS8)
      .def_classmethod(
          "get",
          [](nb::object cls, MlirContext ctxt) {
            return cls(rtgtestRegS8AttrGet(ctxt));
          },
          nb::arg("self"), nb::arg("ctxt") = nullptr);

  mlir_attribute_subclass(m, "RegS9Attr", rtgtestAttrIsARegS9)
      .def_classmethod(
          "get",
          [](nb::object cls, MlirContext ctxt) {
            return cls(rtgtestRegS9AttrGet(ctxt));
          },
          nb::arg("self"), nb::arg("ctxt") = nullptr);

  mlir_attribute_subclass(m, "RegS10Attr", rtgtestAttrIsARegS10)
      .def_classmethod(
          "get",
          [](nb::object cls, MlirContext ctxt) {
            return cls(rtgtestRegS10AttrGet(ctxt));
          },
          nb::arg("self"), nb::arg("ctxt") = nullptr);

  mlir_attribute_subclass(m, "RegS11Attr", rtgtestAttrIsARegS11)
      .def_classmethod(
          "get",
          [](nb::object cls, MlirContext ctxt) {
            return cls(rtgtestRegS11AttrGet(ctxt));
          },
          nb::arg("self"), nb::arg("ctxt") = nullptr);

  mlir_attribute_subclass(m, "RegT3Attr", rtgtestAttrIsARegT3)
      .def_classmethod(
          "get",
          [](nb::object cls, MlirContext ctxt) {
            return cls(rtgtestRegT3AttrGet(ctxt));
          },
          nb::arg("self"), nb::arg("ctxt") = nullptr);

  mlir_attribute_subclass(m, "RegT4Attr", rtgtestAttrIsARegT4)
      .def_classmethod(
          "get",
          [](nb::object cls, MlirContext ctxt) {
            return cls(rtgtestRegT4AttrGet(ctxt));
          },
          nb::arg("self"), nb::arg("ctxt") = nullptr);

  mlir_attribute_subclass(m, "RegT5Attr", rtgtestAttrIsARegT5)
      .def_classmethod(
          "get",
          [](nb::object cls, MlirContext ctxt) {
            return cls(rtgtestRegT5AttrGet(ctxt));
          },
          nb::arg("self"), nb::arg("ctxt") = nullptr);

  mlir_attribute_subclass(m, "RegT6Attr", rtgtestAttrIsARegT6)
      .def_classmethod(
          "get",
          [](nb::object cls, MlirContext ctxt) {
            return cls(rtgtestRegT6AttrGet(ctxt));
          },
          nb::arg("self"), nb::arg("ctxt") = nullptr);
}
