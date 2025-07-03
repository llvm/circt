//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "CIRCTModules.h"

#include "circt-c/Dialect/AIG.h"

#include "mlir/Bindings/Python/NanobindAdaptors.h"

#include "NanobindUtils.h"
#include <nanobind/nanobind.h>
namespace nb = nanobind;

using namespace circt;
using namespace mlir::python::nanobind_adaptors;

/// Populate the aig python module.
void circt::python::populateDialectAIGSubmodule(nb::module_ &m) {
  m.doc() = "AIG dialect Python native extension";
}
