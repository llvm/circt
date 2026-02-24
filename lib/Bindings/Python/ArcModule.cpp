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
#include "mlir/Bindings/Python/IRCore.h"

#include <nanobind/nanobind.h>
namespace nb = nanobind;

using namespace circt;
using mlir::python::MLIR_BINDINGS_PYTHON_DOMAIN::DefaultingPyMlirContext;
using mlir::python::MLIR_BINDINGS_PYTHON_DOMAIN::PyConcreteType;
using mlir::python::MLIR_BINDINGS_PYTHON_DOMAIN::PyMlirContext;

struct PyStateType : PyConcreteType<PyStateType> {
  static constexpr IsAFunctionTy isaFunction = arcTypeIsAState;
  static constexpr const char *pyClassName = "StateType";
  using Base::Base;

  static void bindDerived(ClassTy &c) {
    c.def_static("get",
                 [](MlirType innerType) {
                   auto type = arcStateTypeGet(innerType);
                   return PyStateType(
                       PyMlirContext::forContext(mlirTypeGetContext(type)),
                       type);
                 },
                 nb::arg("inner_type"));
    c.def_prop_ro(
        "type", [](PyStateType &self) { return arcStateTypeGetType(self); });
    c.def_prop_ro("bit_width", [](PyStateType &self) {
      return arcStateTypeGetBitWidth(self);
    });
    c.def_prop_ro("byte_width", [](PyStateType &self) {
      return arcStateTypeGetByteWidth(self);
    });
  }
};

struct PyMemoryType : PyConcreteType<PyMemoryType> {
  static constexpr IsAFunctionTy isaFunction = arcTypeIsAMemory;
  static constexpr const char *pyClassName = "MemoryType";
  using Base::Base;

  static void bindDerived(ClassTy &c) {
    c.def_static(
        "get",
        [](unsigned numWords, MlirType wordType, MlirType addressType) {
          auto type = arcMemoryTypeGet(numWords, wordType, addressType);
          return PyMemoryType(
              PyMlirContext::forContext(mlirTypeGetContext(type)), type);
        },
        nb::arg("num_words"), nb::arg("word_type"), nb::arg("address_type"));
    c.def_prop_ro("num_words", [](PyMemoryType &self) {
      return arcMemoryTypeGetNumWords(self);
    });
    c.def_prop_ro("word_type", [](PyMemoryType &self) {
      return arcMemoryTypeGetWordType(self);
    });
    c.def_prop_ro("address_type", [](PyMemoryType &self) {
      return arcMemoryTypeGetAddressType(self);
    });
    c.def_prop_ro(
        "stride",
        [](PyMemoryType &self) { return arcMemoryTypeGetStride(self); });
  }
};

struct PyStorageType : PyConcreteType<PyStorageType> {
  static constexpr IsAFunctionTy isaFunction = arcTypeIsAStorage;
  static constexpr const char *pyClassName = "StorageType";
  using Base::Base;

  static void bindDerived(ClassTy &c) {
    c.def_static(
        "get",
        [](DefaultingPyMlirContext ctx, nb::object size) {
          MlirType type;
          if (size.is_none())
            type = arcStorageTypeGet(ctx->get());
          else
            type = arcStorageTypeGetWithSize(ctx->get(),
                                            nb::cast<unsigned>(size));
          return PyStorageType(ctx->getRef(), type);
        },
        nb::arg("context").none() = nb::none(),
        nb::arg("size") = nb::none());
    c.def_prop_ro("size", [](PyStorageType &self) {
      return arcStorageTypeGetSize(self);
    });
  }
};

struct PySimModelInstanceType : PyConcreteType<PySimModelInstanceType> {
  static constexpr IsAFunctionTy isaFunction = arcTypeIsASimModelInstance;
  static constexpr const char *pyClassName = "SimModelInstanceType";
  using Base::Base;

  static void bindDerived(ClassTy &c) {
    c.def_static(
        "get",
        [](MlirAttribute model) {
          auto type = arcSimModelInstanceTypeGet(model);
          return PySimModelInstanceType(
              PyMlirContext::forContext(mlirTypeGetContext(type)), type);
        },
        nb::arg("model"));
    c.def_prop_ro("model", [](PySimModelInstanceType &self) {
      return arcSimModelInstanceTypeGetModel(self);
    });
  }
};

void circt::python::populateDialectArcSubmodule(nb::module_ &m) {
  m.doc() = "Arc dialect Python native extension";

  PyStateType::bind(m);
  PyMemoryType::bind(m);
  PyStorageType::bind(m);
  PySimModelInstanceType::bind(m);
}
