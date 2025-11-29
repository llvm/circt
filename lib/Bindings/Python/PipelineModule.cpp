//===- PipelineModule.cpp - Pipeline API nanobind module ------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "CIRCTModules.h"
#include <nanobind/nanobind.h>

namespace nb = nanobind;

void circt::python::populateDialectPipelineSubmodule(nb::module_ &m) {
  m.doc() = "Pipeline dialect Python native extension";
}
