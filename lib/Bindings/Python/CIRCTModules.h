//===- CIRCTModules.h - Populate submodules -------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Functions to populate submodules in CIRCT (if provided).
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_BINDINGS_PYTHON_CIRCTMODULES_H
#define CIRCT_BINDINGS_PYTHON_CIRCTMODULES_H

#include <nanobind/nanobind.h>

namespace circt {
namespace python {

void populateDialectArcSubmodule(nanobind::module_ &m);
void populateDialectESISubmodule(nanobind::module_ &m);
void populateDialectHWSubmodule(nanobind::module_ &m);
void populateDialectMSFTSubmodule(nanobind::module_ &m);
void populateDialectOMSubmodule(nanobind::module_ &m);
void populateDialectRTGSubmodule(nanobind::module_ &m);
#ifdef CIRCT_INCLUDE_TESTS
void populateDialectRTGTestSubmodule(nanobind::module_ &m);
#endif
void populateDialectSeqSubmodule(nanobind::module_ &m);
void populateDialectSVSubmodule(nanobind::module_ &m);
void populateDialectSynthSubmodule(nanobind::module_ &m);
void populateSupportSubmodule(nanobind::module_ &m);

} // namespace python
} // namespace circt

#endif // CIRCT_BINDINGS_PYTHON_DIALECTMODULES_H
