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
#include "esi/StdServices.h"

// pybind11 includes
#include "pybind11/pybind11.h"
namespace py = pybind11;

using namespace esi;
using namespace esi::services;

// NOLINTNEXTLINE(readability-identifier-naming)
PYBIND11_MODULE(esiCppAccel, m) {
  py::class_<Accelerator>(m, "Accelerator")
      .def(py::init(&registry::connect))
      .def("sysinfo", &Accelerator::getService<SysInfo>,
           py::return_value_policy::reference_internal)
      .def("get_service_mmio", &Accelerator::getService<services::MMIO>,
           py::return_value_policy::reference_internal);

  py::class_<SysInfo>(m, "SysInfo")
      .def("esi_version", &SysInfo::esiVersion)
      .def("json_manifest", &SysInfo::jsonManifest);

  py::class_<services::MMIO>(m, "MMIO")
      .def("read", &services::MMIO::read)
      .def("write", &services::MMIO::write);
}
