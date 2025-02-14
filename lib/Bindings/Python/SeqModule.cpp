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
}
