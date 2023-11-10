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
#include "esi/Design.h"
#include "esi/StdServices.h"

#include <sstream>

// pybind11 includes
#include <pybind11/pybind11.h>
namespace py = pybind11;

#include <pybind11/stl.h>

using namespace esi;
using namespace esi::services;

// NOLINTNEXTLINE(readability-identifier-naming)
PYBIND11_MODULE(esiCppAccel, m) {
  py::class_<ModuleInfo>(m, "ModuleInfo")
      .def_property_readonly("name", [](ModuleInfo &info) { return info.name; })
      .def_property_readonly("summary",
                             [](ModuleInfo &info) { return info.summary; })
      .def_property_readonly("version",
                             [](ModuleInfo &info) { return info.version; })
      .def_property_readonly("repo", [](ModuleInfo &info) { return info.repo; })
      .def_property_readonly("commit_hash",
                             [](ModuleInfo &info) { return info.commitHash; })
      // TODO: "extra" field.
      .def("__repr__", [](ModuleInfo &info) {
        std::string ret;
        std::stringstream os(ret);
        os << info;
        return os.str();
      });

  py::class_<SysInfo>(m, "SysInfo")
      .def("esi_version", &SysInfo::getEsiVersion)
      .def("json_manifest", &SysInfo::getJsonManifest);

  py::class_<services::MMIO>(m, "MMIO")
      .def("read", &services::MMIO::read)
      .def("write", &services::MMIO::write);

  py::class_<AppID>(m, "AppID")
      .def_property_readonly("name", [](AppID &id) { return id.name; })
      .def_property_readonly("idx",
                             [](AppID &id) -> py::object {
                               if (id.idx)
                                 return py::cast(id.idx);
                               return py::none();
                             })
      .def("__repr__", [](AppID &id) {
        std::string ret = "<" + id.name;
        if (id.idx)
          ret = ret + "[" + std::to_string(*id.idx) + "]";
        ret = ret + ">";
        return ret;
      });

  py::class_<ChannelPort>(m, "ChannelPort");
  py::class_<WriteChannelPort, ChannelPort>(m, "WriteChannelPort")
      .def("write", [](WriteChannelPort &p, std::vector<uint8_t> data) {
        p.write(data.data(), data.size());
      });
  py::class_<ReadChannelPort, ChannelPort>(m, "ReadChannelPort")
      .def("read", [](ReadChannelPort &p, size_t maxSize) {
        std::vector<uint8_t> data(maxSize);
        ssize_t size = p.read(data.data(), data.size());
        if (size < 0)
          throw std::runtime_error("read failed");
        data.resize(size);
        return data;
      });

  py::class_<BundlePort>(m, "BundlePort")
      .def("getWrite", &BundlePort::getRawWrite,
           py::return_value_policy::reference)
      .def("getRead", &BundlePort::getRawRead,
           py::return_value_policy::reference);

  // Store this variable (not commonly done) as the "children" method needs for
  // "Instance" to be defined first.
  auto design =
      py::class_<Design>(m, "Design")
          .def_property_readonly("info", &Design::getInfo)
          .def_property_readonly("ports", &Design::getPorts,
                                 py::return_value_policy::reference_internal);

  // In order to inherit methods from "Design", it needs to be defined first.
  py::class_<Instance, Design>(m, "Instance")
      .def_property_readonly("id", &Instance::getID);

  // Since this returns a vector of Instance*, we need to define Instance first
  // or else pybind11-stubgen complains.
  design.def_property_readonly(
      "children",
      [](Design &d) -> std::vector<Instance *> {
        std::vector<Instance *> ret;
        for (auto &c : d.getChildren())
          ret.push_back(c.get());
        return ret;
      },
      py::return_value_policy::reference);

  py::class_<Accelerator>(m, "Accelerator")
      .def(py::init(&registry::connect))
      .def(
          "sysinfo",
          [](Accelerator &acc) {
            return acc.getService<services::SysInfo>({});
          },
          py::return_value_policy::reference_internal)
      .def(
          "get_service_mmio",
          [](Accelerator &acc) { return acc.getService<services::MMIO>({}); },
          py::return_value_policy::reference_internal);

  py::class_<Type>(m, "Type")
      .def_property_readonly("id", &Type::getID)
      .def("__repr__", [](Type &t) { return "<" + t.getID() + ">"; });

  py::class_<Manifest>(m, "Manifest")
      .def(py::init<std::string>())
      .def_property_readonly("api_version", &Manifest::getApiVersion)
      .def("build_design", &Manifest::buildDesign)
      .def_property_readonly("type_table", &Manifest::getTypeTable);
}
