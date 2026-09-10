//===- SeqModule.cpp - Seq API nanobind module ----------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "CIRCTModules.h"

#include "circt-c/Dialect/Seq.h"
#include "mlir/Bindings/Python/NanobindAdaptors.h"

#include "NanobindUtils.h"
#include "mlir-c/Support.h"
#include <nanobind/nanobind.h>

namespace nb = nanobind;

using namespace circt;
using namespace mlir::python::nanobind_adaptors;

/// Populate the seq python module.
void circt::python::populateDialectSeqSubmodule(nb::module_ &m) {
  m.doc() = "Seq dialect Python native extension";

  mlir_type_subclass(m, "ClockType", seqTypeIsAClock)
      .def_classmethod(
          "get",
          [](nb::object cls, MlirContext ctx) {
            return cls(seqClockTypeGet(ctx));
          },
          nb::arg("cls"), nb::arg("context") = nb::none());

  mlir_type_subclass(m, "ImmutableType", seqTypeIsAImmutable)
      .def_classmethod("get",
                       [](nb::object cls, MlirType innerType) {
                         return cls(seqImmutableTypeGet(innerType));
                       })
      .def_property_readonly("inner_type", [](MlirType self) {
        return seqImmutableTypeGetInnerType(self);
      });

  mlir_type_subclass(m, "HLMemType", seqTypeIsAHLMem)
      .def_classmethod(
          "get",
          [](nb::object cls, std::vector<int64_t> shape, MlirType elementType,
             MlirContext ctx) {
            return cls(
                seqHLMemTypeGet(ctx, shape.size(), shape.data(), elementType));
          },
          nb::arg("cls"), nb::arg("shape"), nb::arg("element_type"),
          nb::arg("context") = nb::none())
      .def_property_readonly(
          "element_type",
          [](MlirType self) { return seqHLMemTypeGetElementType(self); })
      .def_property_readonly(
          "rank", [](MlirType self) { return seqHLMemTypeGetRank(self); })
      .def_property_readonly("shape", [](MlirType self) {
        intptr_t rank = seqHLMemTypeGetRank(self);
        const int64_t *shapePtr = seqHLMemTypeGetShape(self);
        nb::list result;
        for (intptr_t i = 0; i < rank; ++i)
          result.append(shapePtr[i]);
        return result;
      });

  mlir_type_subclass(m, "FirMemType", seqTypeIsAFirMem)
      .def_classmethod(
          "get",
          [](nb::object cls, uint64_t depth, uint32_t width,
             std::optional<uint32_t> maskWidth, MlirContext ctx) {
            const uint32_t *maskPtr = nullptr;
            uint32_t maskVal = 0;
            if (maskWidth.has_value()) {
              maskVal = maskWidth.value();
              maskPtr = &maskVal;
            }
            return cls(seqFirMemTypeGet(ctx, depth, width, maskPtr));
          },
          nb::arg("cls"), nb::arg("depth"), nb::arg("width"),
          nb::arg("mask_width") = nb::none(), nb::arg("context") = nb::none())
      .def_property_readonly(
          "depth", [](MlirType self) { return seqFirMemTypeGetDepth(self); })
      .def_property_readonly(
          "width", [](MlirType self) { return seqFirMemTypeGetWidth(self); })
      .def_property_readonly("mask_width", [](MlirType self) -> nb::object {
        if (seqFirMemTypeHasMask(self))
          return nb::cast(seqFirMemTypeGetMaskWidth(self));
        return nb::none();
      });
}
