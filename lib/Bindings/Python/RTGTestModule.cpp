//===- RTGTestModule.cpp - RTGTest API nanobind module --------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "CIRCTModules.h"

#include "circt-c/Dialect/RTGTest.h"

#include "mlir/Bindings/Python/IRCore.h"

#include <nanobind/nanobind.h>
namespace nb = nanobind;

using namespace circt;
using mlir::python::MLIR_BINDINGS_PYTHON_DOMAIN::DefaultingPyMlirContext;
using mlir::python::MLIR_BINDINGS_PYTHON_DOMAIN::PyConcreteAttribute;
using mlir::python::MLIR_BINDINGS_PYTHON_DOMAIN::PyConcreteType;

struct PyCPUType : PyConcreteType<PyCPUType> {
  static constexpr IsAFunctionTy isaFunction = rtgtestTypeIsACPU;
  static constexpr const char *pyClassName = "CPUType";
  using Base::Base;

  static void bindDerived(ClassTy &c) {
    c.def_static(
        "get",
        [](DefaultingPyMlirContext ctx) {
          return PyCPUType(ctx->getRef(), rtgtestCPUTypeGet(ctx->get()));
        },
        nb::arg("ctxt").none() = nb::none());
  }
};

struct PyIntegerRegisterType : PyConcreteType<PyIntegerRegisterType> {
  static constexpr IsAFunctionTy isaFunction = rtgtestTypeIsAIntegerRegister;
  static constexpr const char *pyClassName = "IntegerRegisterType";
  using Base::Base;

  static void bindDerived(ClassTy &c) {
    c.def_static(
        "get",
        [](DefaultingPyMlirContext ctx) {
          return PyIntegerRegisterType(ctx->getRef(),
                                       rtgtestIntegerRegisterTypeGet(ctx->get()));
        },
        nb::arg("ctxt").none() = nb::none());
  }
};

struct PyCPUAttr : PyConcreteAttribute<PyCPUAttr> {
  static constexpr IsAFunctionTy isaFunction = rtgtestAttrIsACPU;
  static constexpr const char *pyClassName = "CPUAttr";
  using Base::Base;

  static void bindDerived(ClassTy &c) {
    c.def_static(
        "get",
        [](unsigned id, DefaultingPyMlirContext ctx) {
          return PyCPUAttr(ctx->getRef(),
                           rtgtestCPUAttrGet(ctx->get(), id));
        },
        nb::arg("id"), nb::arg("ctxt").none() = nb::none());
    c.def_prop_ro(
        "id", [](PyCPUAttr &self) { return rtgtestCPUAttrGetId(self); });
  }
};

#define DEFINE_SIMPLE_REG_ATTR(PyName, CName, isaFn, getFn)                    \
  struct PyName : PyConcreteAttribute<PyName> {                                \
    static constexpr IsAFunctionTy isaFunction = isaFn;                        \
    static constexpr const char *pyClassName = CName;                          \
    using Base::Base;                                                          \
                                                                               \
    static void bindDerived(ClassTy &c) {                                      \
      c.def_static(                                                            \
          "get",                                                               \
          [](DefaultingPyMlirContext ctx) {                                     \
            return PyName(ctx->getRef(), getFn(ctx->get()));                    \
          },                                                                   \
          nb::arg("ctxt").none() = nb::none());                                \
    }                                                                          \
  };

DEFINE_SIMPLE_REG_ATTR(PyRegZeroAttr, "RegZeroAttr", rtgtestAttrIsARegZero,
                       rtgtestRegZeroAttrGet)
DEFINE_SIMPLE_REG_ATTR(PyRegRaAttr, "RegRaAttr", rtgtestAttrIsARegRa,
                       rtgtestRegRaAttrGet)
DEFINE_SIMPLE_REG_ATTR(PyRegSpAttr, "RegSpAttr", rtgtestAttrIsARegSp,
                       rtgtestRegSpAttrGet)
DEFINE_SIMPLE_REG_ATTR(PyRegGpAttr, "RegGpAttr", rtgtestAttrIsARegGp,
                       rtgtestRegGpAttrGet)
DEFINE_SIMPLE_REG_ATTR(PyRegTpAttr, "RegTpAttr", rtgtestAttrIsARegTp,
                       rtgtestRegTpAttrGet)
DEFINE_SIMPLE_REG_ATTR(PyRegT0Attr, "RegT0Attr", rtgtestAttrIsARegT0,
                       rtgtestRegT0AttrGet)
DEFINE_SIMPLE_REG_ATTR(PyRegT1Attr, "RegT1Attr", rtgtestAttrIsARegT1,
                       rtgtestRegT1AttrGet)
DEFINE_SIMPLE_REG_ATTR(PyRegT2Attr, "RegT2Attr", rtgtestAttrIsARegT2,
                       rtgtestRegT2AttrGet)
DEFINE_SIMPLE_REG_ATTR(PyRegS0Attr, "RegS0Attr", rtgtestAttrIsARegS0,
                       rtgtestRegS0AttrGet)
DEFINE_SIMPLE_REG_ATTR(PyRegS1Attr, "RegS1Attr", rtgtestAttrIsARegS1,
                       rtgtestRegS1AttrGet)
DEFINE_SIMPLE_REG_ATTR(PyRegA0Attr, "RegA0Attr", rtgtestAttrIsARegA0,
                       rtgtestRegA0AttrGet)
DEFINE_SIMPLE_REG_ATTR(PyRegA1Attr, "RegA1Attr", rtgtestAttrIsARegA1,
                       rtgtestRegA1AttrGet)
DEFINE_SIMPLE_REG_ATTR(PyRegA2Attr, "RegA2Attr", rtgtestAttrIsARegA2,
                       rtgtestRegA2AttrGet)
DEFINE_SIMPLE_REG_ATTR(PyRegA3Attr, "RegA3Attr", rtgtestAttrIsARegA3,
                       rtgtestRegA3AttrGet)
DEFINE_SIMPLE_REG_ATTR(PyRegA4Attr, "RegA4Attr", rtgtestAttrIsARegA4,
                       rtgtestRegA4AttrGet)
DEFINE_SIMPLE_REG_ATTR(PyRegA5Attr, "RegA5Attr", rtgtestAttrIsARegA5,
                       rtgtestRegA5AttrGet)
DEFINE_SIMPLE_REG_ATTR(PyRegA6Attr, "RegA6Attr", rtgtestAttrIsARegA6,
                       rtgtestRegA6AttrGet)
DEFINE_SIMPLE_REG_ATTR(PyRegA7Attr, "RegA7Attr", rtgtestAttrIsARegA7,
                       rtgtestRegA7AttrGet)
DEFINE_SIMPLE_REG_ATTR(PyRegS2Attr, "RegS2Attr", rtgtestAttrIsARegS2,
                       rtgtestRegS2AttrGet)
DEFINE_SIMPLE_REG_ATTR(PyRegS3Attr, "RegS3Attr", rtgtestAttrIsARegS3,
                       rtgtestRegS3AttrGet)
DEFINE_SIMPLE_REG_ATTR(PyRegS4Attr, "RegS4Attr", rtgtestAttrIsARegS4,
                       rtgtestRegS4AttrGet)
DEFINE_SIMPLE_REG_ATTR(PyRegS5Attr, "RegS5Attr", rtgtestAttrIsARegS5,
                       rtgtestRegS5AttrGet)
DEFINE_SIMPLE_REG_ATTR(PyRegS6Attr, "RegS6Attr", rtgtestAttrIsARegS6,
                       rtgtestRegS6AttrGet)
DEFINE_SIMPLE_REG_ATTR(PyRegS7Attr, "RegS7Attr", rtgtestAttrIsARegS7,
                       rtgtestRegS7AttrGet)
DEFINE_SIMPLE_REG_ATTR(PyRegS8Attr, "RegS8Attr", rtgtestAttrIsARegS8,
                       rtgtestRegS8AttrGet)
DEFINE_SIMPLE_REG_ATTR(PyRegS9Attr, "RegS9Attr", rtgtestAttrIsARegS9,
                       rtgtestRegS9AttrGet)
DEFINE_SIMPLE_REG_ATTR(PyRegS10Attr, "RegS10Attr", rtgtestAttrIsARegS10,
                       rtgtestRegS10AttrGet)
DEFINE_SIMPLE_REG_ATTR(PyRegS11Attr, "RegS11Attr", rtgtestAttrIsARegS11,
                       rtgtestRegS11AttrGet)
DEFINE_SIMPLE_REG_ATTR(PyRegT3Attr, "RegT3Attr", rtgtestAttrIsARegT3,
                       rtgtestRegT3AttrGet)
DEFINE_SIMPLE_REG_ATTR(PyRegT4Attr, "RegT4Attr", rtgtestAttrIsARegT4,
                       rtgtestRegT4AttrGet)
DEFINE_SIMPLE_REG_ATTR(PyRegT5Attr, "RegT5Attr", rtgtestAttrIsARegT5,
                       rtgtestRegT5AttrGet)
DEFINE_SIMPLE_REG_ATTR(PyRegT6Attr, "RegT6Attr", rtgtestAttrIsARegT6,
                       rtgtestRegT6AttrGet)

#undef DEFINE_SIMPLE_REG_ATTR

/// Populate the rtgtest python module.
void circt::python::populateDialectRTGTestSubmodule(nb::module_ &m) {
  m.doc() = "RTGTest dialect Python native extension";

  PyCPUType::bind(m);
  PyIntegerRegisterType::bind(m);
  PyCPUAttr::bind(m);
  PyRegZeroAttr::bind(m);
  PyRegRaAttr::bind(m);
  PyRegSpAttr::bind(m);
  PyRegGpAttr::bind(m);
  PyRegTpAttr::bind(m);
  PyRegT0Attr::bind(m);
  PyRegT1Attr::bind(m);
  PyRegT2Attr::bind(m);
  PyRegS0Attr::bind(m);
  PyRegS1Attr::bind(m);
  PyRegA0Attr::bind(m);
  PyRegA1Attr::bind(m);
  PyRegA2Attr::bind(m);
  PyRegA3Attr::bind(m);
  PyRegA4Attr::bind(m);
  PyRegA5Attr::bind(m);
  PyRegA6Attr::bind(m);
  PyRegA7Attr::bind(m);
  PyRegS2Attr::bind(m);
  PyRegS3Attr::bind(m);
  PyRegS4Attr::bind(m);
  PyRegS5Attr::bind(m);
  PyRegS6Attr::bind(m);
  PyRegS7Attr::bind(m);
  PyRegS8Attr::bind(m);
  PyRegS9Attr::bind(m);
  PyRegS10Attr::bind(m);
  PyRegS11Attr::bind(m);
  PyRegT3Attr::bind(m);
  PyRegT4Attr::bind(m);
  PyRegT5Attr::bind(m);
  PyRegT6Attr::bind(m);
}
