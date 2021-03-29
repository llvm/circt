//===- RTLModule.h - RTL Submodule of pybind module -----------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_BINDINGS_PYTHON_RTLMODULE_H
#define CIRCT_BINDINGS_PYTHON_RTLMODULE_H

#include "pybind11/pybind11.h"

void populateRTLModule(pybind11::module &m);

#endif // CIRCT_BINDINGS_PYTHON_RTLMODULE_H
