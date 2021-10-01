//===- MSFTModule.cpp - MSFT API pybind module ----------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "DialectModules.h"

#include "circt-c/Dialect/MSFT.h"
#include "circt/Support/LLVM.h"

#include "mlir/Bindings/Python/PybindAdaptors.h"
#include "mlir/CAPI/IR.h"
#include "mlir/CAPI/Support.h"

#include "PybindUtils.h"
#include <pybind11/functional.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
namespace py = pybind11;

using namespace circt;
using namespace circt::msft;
using namespace mlir::python::adaptors;

class DeviceDB {
public:
  DeviceDB(MlirOperation top) { db = circtMSFTCreateDeviceDB(top); }
  size_t addDesignPlacements() {
    return circtMSFTDeviceDBAddDesignPlacements(db);
  }
  bool addPlacement(MlirAttribute loc, MlirAttribute path, std::string subpath,
                    MlirOperation op) {
    return mlirLogicalResultIsSuccess(circtMSFTDeviceDBAddPlacement(
        db, loc,
        CirctMSFTPlacedInstance{path, subpath.c_str(), subpath.size(), op}));
  }
  py::object getInstanceAt(MlirAttribute loc) {
    CirctMSFTPlacedInstance inst;
    if (!circtMSFTDeviceDBTryGetInstanceAt(db, loc, &inst))
      return py::none();
    std::string subpath(inst.subpath, inst.subpathLength);
    return (py::tuple)py::cast(std::make_tuple(inst.path, subpath, inst.op));
  }

private:
  CirctMSFTDeviceDB db;
};

/// Populate the msft python module.
void circt::python::populateDialectMSFTSubmodule(py::module &m) {
  mlirMSFTRegisterPasses();

  m.doc() = "MSFT dialect Python native extension";

  m.def("get_instance", circtMSFTGetInstance, py::arg("root"), py::arg("path"));

  py::enum_<DeviceType>(m, "DeviceType")
      .value("M20K", DeviceType::M20K)
      .value("DSP", DeviceType::DSP)
      .export_values();

  m.def("export_tcl", [](MlirOperation mod, py::object fileObject) {
    circt::python::PyFileAccumulator accum(fileObject, false);
    py::gil_scoped_release();
    mlirMSFTExportTcl(mod, accum.getCallback(), accum.getUserData());
  });

  mlir_attribute_subclass(m, "PhysLocationAttr",
                          circtMSFTAttributeIsAPhysLocationAttribute)
      .def_classmethod(
          "get",
          [](py::object cls, DeviceType devType, uint64_t x, uint64_t y,
             uint64_t num, MlirContext ctxt) {
            return cls(circtMSFTPhysLocationAttrGet(ctxt, (uint64_t)devType, x,
                                                    y, num));
          },
          "Create a physical location attribute", py::arg(),
          py::arg("dev_type"), py::arg("x"), py::arg("y"), py::arg("num"),
          py::arg("ctxt") = py::none())
      .def_property_readonly(
          "devtype",
          [](MlirAttribute self) {
            return (DeviceType)circtMSFTPhysLocationAttrGetDeviceType(self);
          })
      .def_property_readonly("x",
                             [](MlirAttribute self) {
                               return (DeviceType)circtMSFTPhysLocationAttrGetX(
                                   self);
                             })
      .def_property_readonly("y",
                             [](MlirAttribute self) {
                               return (DeviceType)circtMSFTPhysLocationAttrGetY(
                                   self);
                             })
      .def_property_readonly("num", [](MlirAttribute self) {
        return (DeviceType)circtMSFTPhysLocationAttrGetNum(self);
      });

  mlir_attribute_subclass(m, "RootedInstancePathAttr",
                          circtMSFTAttributeIsARootedInstancePathAttribute)
      .def_classmethod(
          "get",
          [](py::object cls, MlirAttribute rootSymbol,
             std::vector<MlirAttribute> instancePath, MlirContext ctxt) {
            return cls(circtMSFTRootedInstancePathAttrGet(
                ctxt, rootSymbol, instancePath.data(), instancePath.size()));
          },
          "Create an rooted instance path attribute", py::arg(),
          py::arg("root_symbol"), py::arg("instance_path"),
          py::arg("ctxt") = py::none());

  mlir_attribute_subclass(m, "SwitchInstanceAttr",
                          circtMSFTAttributeIsASwitchInstanceAttribute)
      .def_classmethod(
          "get",
          [](py::object cls,
             std::vector<std::pair<MlirAttribute, MlirAttribute>> listOfCases,
             MlirContext ctxt) {
            std::vector<CirctMSFTSwitchInstanceCase> cases;
            for (auto p : listOfCases)
              cases.push_back({std::get<0>(p), std::get<1>(p)});
            return cls(circtMSFTSwitchInstanceAttrGet(ctxt, cases.data(),
                                                      cases.size()));
          },
          "Create an instance switch attribute", py::arg(),
          py::arg("list_of_cases"), py::arg("ctxt") = py::none())
      .def_property_readonly(
          "cases",
          [](MlirAttribute self) {
            size_t numCases = circtMSFTSwitchInstanceAttrGetNumCases(self);
            std::vector<CirctMSFTSwitchInstanceCase> cases(numCases);
            circtMSFTSwitchInstanceAttrGetCases(self, cases.data(),
                                                cases.max_size());
            std::vector<std::pair<MlirAttribute, MlirAttribute>> pyCases;
            for (auto c : cases)
              pyCases.push_back(std::make_pair(c.instance, c.attr));
            return pyCases;
          })
      .def_property_readonly("num_cases", [](MlirAttribute self) {
        return circtMSFTSwitchInstanceAttrGetNumCases(self);
      });

  py::class_<DeviceDB>(m, "DeviceDB")
      .def(py::init<MlirOperation>(), py::arg("top"))
      .def("add_design_placements", &DeviceDB::addDesignPlacements,
           "Add the placements already present in the design.")
      .def("add_placement", &DeviceDB::addPlacement,
           "Inform the DB about a new placement.", py::arg("location"),
           py::arg("path"), py::arg("subpath"), py::arg("op"))
      .def("get_instance_at", &DeviceDB::getInstanceAt,
           "Get the instance at location. Returns None if nothing exists "
           "there. Otherwise, returns (path, subpath, op) of the instance "
           "there.");
}
