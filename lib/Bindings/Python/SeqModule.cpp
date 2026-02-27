//===- SeqModule.cpp - Seq API nanobind module ----------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "CIRCTModules.h"

#include "circt-c/Dialect/Seq.h"
#include "mlir/Bindings/Python/IRCore.h"

#include "NanobindUtils.h"
#include "mlir-c/Support.h"
#include <nanobind/nanobind.h>

namespace nb = nanobind;

using namespace circt;
using mlir::python::MLIR_BINDINGS_PYTHON_DOMAIN::DefaultingPyMlirContext;
using mlir::python::MLIR_BINDINGS_PYTHON_DOMAIN::PyConcreteType;
using mlir::python::MLIR_BINDINGS_PYTHON_DOMAIN::PyMlirContext;

struct PyClockType : PyConcreteType<PyClockType> {
  static constexpr IsAFunctionTy isaFunction = seqTypeIsAClock;
  static constexpr GetTypeIDFunctionTy getTypeIdFunction = seqClockTypeGetTypeID;
  static constexpr const char *pyClassName = "ClockType";
  using Base::Base;

  static void bindDerived(ClassTy &c) {
    c.def_static(
        "get",
        [](DefaultingPyMlirContext ctx) {
          return PyClockType(ctx->getRef(), seqClockTypeGet(ctx->get()));
        },
        nb::arg("context").none() = nb::none());
  }
};

struct PyImmutableType : PyConcreteType<PyImmutableType> {
  static constexpr IsAFunctionTy isaFunction = seqTypeIsAImmutable;
  static constexpr const char *pyClassName = "ImmutableType";
  using Base::Base;

  static void bindDerived(ClassTy &c) {
    c.def_static("get", [](MlirType innerType) {
      auto type = seqImmutableTypeGet(innerType);
      return PyImmutableType(
          PyMlirContext::forContext(mlirTypeGetContext(type)), type);
    });
    c.def_prop_ro("inner_type", [](PyImmutableType &self) {
      return seqImmutableTypeGetInnerType(self);
    });
  }
};

struct PyHLMemType : PyConcreteType<PyHLMemType> {
  static constexpr IsAFunctionTy isaFunction = seqTypeIsAHLMem;
  static constexpr const char *pyClassName = "HLMemType";
  using Base::Base;

  static void bindDerived(ClassTy &c) {
    c.def_static(
        "get",
        [](std::vector<int64_t> shape, MlirType elementType,
           DefaultingPyMlirContext ctx) {
          return PyHLMemType(ctx->getRef(),
                             seqHLMemTypeGet(ctx->get(), shape.size(),
                                             shape.data(), elementType));
        },
        nb::arg("shape"), nb::arg("element_type"),
        nb::arg("context").none() = nb::none());
    c.def_prop_ro("element_type", [](PyHLMemType &self) {
      return seqHLMemTypeGetElementType(self);
    });
    c.def_prop_ro("rank",
                  [](PyHLMemType &self) { return seqHLMemTypeGetRank(self); });
    c.def_prop_ro("shape", [](PyHLMemType &self) {
      intptr_t rank = seqHLMemTypeGetRank(self);
      const int64_t *shapePtr = seqHLMemTypeGetShape(self);
      nb::list result;
      for (intptr_t i = 0; i < rank; ++i)
        result.append(shapePtr[i]);
      return result;
    });
  }
};

struct PyFirMemType : PyConcreteType<PyFirMemType> {
  static constexpr IsAFunctionTy isaFunction = seqTypeIsAFirMem;
  static constexpr const char *pyClassName = "FirMemType";
  using Base::Base;

  static void bindDerived(ClassTy &c) {
    c.def_static(
        "get",
        [](uint64_t depth, uint32_t width, std::optional<uint32_t> maskWidth,
           DefaultingPyMlirContext ctx) {
          const uint32_t *maskPtr = nullptr;
          uint32_t maskVal = 0;
          if (maskWidth.has_value()) {
            maskVal = maskWidth.value();
            maskPtr = &maskVal;
          }
          return PyFirMemType(ctx->getRef(), seqFirMemTypeGet(ctx->get(), depth,
                                                              width, maskPtr));
        },
        nb::arg("depth"), nb::arg("width"), nb::arg("mask_width") = nb::none(),
        nb::arg("context").none() = nb::none());
    c.def_prop_ro("depth", [](PyFirMemType &self) {
      return seqFirMemTypeGetDepth(self);
    });
    c.def_prop_ro("width", [](PyFirMemType &self) {
      return seqFirMemTypeGetWidth(self);
    });
    c.def_prop_ro("mask_width", [](PyFirMemType &self) -> nb::object {
      if (seqFirMemTypeHasMask(self))
        return nb::cast(seqFirMemTypeGetMaskWidth(self));
      return nb::none();
    });
  }
};

/// Populate the seq python module.
void circt::python::populateDialectSeqSubmodule(nb::module_ &m) {
  m.doc() = "Seq dialect Python native extension";

  PyClockType::bind(m);
  PyImmutableType::bind(m);
  PyHLMemType::bind(m);
  PyFirMemType::bind(m);
}
