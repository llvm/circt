//===- esiaccel.cpp - ESI runtime python bindings ---------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Simply wrap the C++ API into a Python module called 'esiaccel'.
//
//===----------------------------------------------------------------------===//

#include "esi/Accelerator.h"

// pybind11 includes
#include "pybind11/pybind11.h"
namespace py = pybind11;

using namespace esi;

// NOLINTNEXTLINE(readability-identifier-naming)
PYBIND11_MODULE(esiaccel, m) {
  py::class_<Accelerator>(m, "Accelerator")
      .def_static("connect", &registry::connect,
                  py::return_value_policy::take_ownership)
      .def("sysinfo", &Accelerator::sysInfo,
           py::return_value_policy::reference_internal);
  py::class_<SysInfo>(m, "SysInfo")
      .def("esi_version", &SysInfo::esiVersion)
      .def("raw_json_manifest", &SysInfo::rawJsonManifest);
}
