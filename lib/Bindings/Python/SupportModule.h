//===- SupportModule.h - Support API submodule ----------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_BINDINGS_PYTHON_SUPPORTMODULE_H
#define CIRCT_BINDINGS_PYTHON_SUPPORTMODULE_H

#include <pybind11/pybind11.h>

namespace circt {
namespace python {

void populateSupportSubmodule(pybind11::module &m);

} // namespace python
} // namespace circt

#endif // CIRCT_BINDINGS_PYTHON_SUPPORTMODULE_H
