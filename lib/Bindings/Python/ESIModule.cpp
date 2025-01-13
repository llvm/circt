//===- ESIModule.cpp - ESI API pybind module ------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "CIRCTModules.h"

#include "circt/Dialect/ESI/ESIDialect.h"

#include "circt-c/Dialect/ESI.h"
#include "mlir-c/Bindings/Python/Interop.h"

#include "mlir/Bindings/Python/PybindAdaptors.h"
#include "mlir/CAPI/IR.h"
#include "mlir/CAPI/Support.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"

#include "PybindUtils.h"
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
namespace py = pybind11;

using namespace circt::esi;

//===----------------------------------------------------------------------===//
// The main entry point into the ESI Assembly API.
//===----------------------------------------------------------------------===//

/// Container for a Python function that will be called to generate a service.
class ServiceGenFunc {
public:
  ServiceGenFunc(py::object genFunc) : genFunc(std::move(genFunc)) {}

  MlirLogicalResult run(MlirOperation reqOp, MlirOperation declOp,
                        MlirOperation recOp) {
    py::gil_scoped_acquire acquire;
    py::object rc = genFunc(reqOp, declOp, recOp);
    return rc.cast<bool>() ? mlirLogicalResultSuccess()
                           : mlirLogicalResultFailure();
  }

private:
  py::object genFunc;
};

// Mapping from unique identifier to python callback. We use std::string
// pointers since we also need to allocate memory for the string.
llvm::DenseMap<std::string *, ServiceGenFunc> serviceGenFuncLookup;
static MlirLogicalResult serviceGenFunc(MlirOperation reqOp,
                                        MlirOperation declOp,
                                        MlirOperation recOp, void *userData) {
  std::string *name = static_cast<std::string *>(userData);
  auto iter = serviceGenFuncLookup.find(name);
  if (iter == serviceGenFuncLookup.end())
    return mlirLogicalResultFailure();
  return iter->getSecond().run(reqOp, declOp, recOp);
}

void registerServiceGenerator(std::string name, py::object genFunc) {
  std::string *n = new std::string(name);
  serviceGenFuncLookup.try_emplace(n, ServiceGenFunc(genFunc));
  circtESIRegisterGlobalServiceGenerator(wrap(*n), serviceGenFunc, n);
}

class PyAppIDIndex {
public:
  PyAppIDIndex(MlirOperation root) { index = circtESIAppIDIndexGet(root); }
  PyAppIDIndex(const PyAppIDIndex &) = delete;
  ~PyAppIDIndex() { circtESIAppIDIndexFree(index); }

  MlirAttribute getChildAppIDsOf(MlirOperation op) const {
    return circtESIAppIDIndexGetChildAppIDsOf(index, op);
  }

  py::object getAppIDPathAttr(MlirOperation fromMod, MlirAttribute appid,
                              MlirLocation loc) const {
    MlirAttribute path =
        circtESIAppIDIndexGetAppIDPath(index, fromMod, appid, loc);
    if (path.ptr == nullptr)
      return py::none();
    return py::cast(path);
  }

private:
  CirctESIAppIDIndex index;
};

using namespace mlir::python::adaptors;

void circt::python::populateDialectESISubmodule(py::module &m) {
  m.doc() = "ESI Python Native Extension";
  ::registerESIPasses();

  // Clean up references when the module is unloaded.
  auto cleanup = []() { serviceGenFuncLookup.clear(); };
  m.def("cleanup", cleanup,
        "Cleanup various references. Must be called before the module is "
        "unloaded in order to not leak.");

  m.def("registerServiceGenerator", registerServiceGenerator,
        "Register a service generator for a given service name.",
        py::arg("impl_type"), py::arg("generator"));

  mlir_type_subclass(m, "ChannelType", circtESITypeIsAChannelType)
      .def_classmethod(
          "get",
          [](py::object cls, MlirType inner, uint32_t signaling = 0,
             uint64_t dataDelay = 0) {
            if (circtESITypeIsAChannelType(inner))
              return cls(inner);
            return cls(circtESIChannelTypeGet(inner, signaling, dataDelay));
          },
          py::arg("cls"), py::arg("inner"), py::arg("signaling") = 0,
          py::arg("dataDelay") = 0)
      .def_property_readonly(
          "inner", [](MlirType self) { return circtESIChannelGetInner(self); })
      .def_property_readonly(
          "signaling",
          [](MlirType self) { return circtESIChannelGetSignaling(self); })
      .def_property_readonly("data_delay", [](MlirType self) {
        return circtESIChannelGetDataDelay(self);
      });

  mlir_type_subclass(m, "AnyType", circtESITypeIsAnAnyType)
      .def_classmethod(
          "get",
          [](py::object cls, MlirContext ctxt) {
            return cls(circtESIAnyTypeGet(ctxt));
          },
          py::arg("self"), py::arg("ctxt") = nullptr);

  mlir_type_subclass(m, "ListType", circtESITypeIsAListType)
      .def_classmethod(
          "get",
          [](py::object cls, MlirType inner) {
            return cls(circtESIListTypeGet(inner));
          },
          py::arg("cls"), py::arg("inner"))
      .def_property_readonly("element_type", [](MlirType self) {
        return circtESIListTypeGetElementType(self);
      });

  py::enum_<ChannelDirection>(m, "ChannelDirection")
      .value("TO", ChannelDirection::to)
      .value("FROM", ChannelDirection::from);
  mlir_type_subclass(m, "BundleType", circtESITypeIsABundleType)
      .def_classmethod(
          "get",
          [](py::object cls, std::vector<py::tuple> channelTuples,
             bool resettable, MlirContext ctxt) {
            llvm::SmallVector<CirctESIBundleTypeBundleChannel, 4> channels(
                llvm::map_range(channelTuples, [ctxt](py::tuple t) {
                  std::string name = py::cast<std::string>(t[0]);
                  return CirctESIBundleTypeBundleChannel{
                      mlirIdentifierGet(ctxt, mlirStringRefCreate(
                                                  name.data(), name.length())),
                      (uint32_t)py::cast<ChannelDirection>(t[1]),
                      py::cast<MlirType>(t[2])};
                }));
            return cls(circtESIBundleTypeGet(ctxt, channels.size(),
                                             channels.data(), resettable));
          },
          py::arg("cls"), py::arg("channels"), py::arg("resettable"),
          py::arg("ctxt") = nullptr)
      .def_property_readonly("resettable", &circtESIBundleTypeGetResettable)
      .def_property_readonly("channels", [](MlirType bundleType) {
        std::vector<py::tuple> channels;
        size_t numChannels = circtESIBundleTypeGetNumChannels(bundleType);
        for (size_t i = 0; i < numChannels; ++i) {
          CirctESIBundleTypeBundleChannel channel =
              circtESIBundleTypeGetChannel(bundleType, i);
          MlirStringRef name = mlirIdentifierStr(channel.name);
          channels.push_back(py::make_tuple(py::str(name.data, name.length),
                                            (ChannelDirection)channel.direction,
                                            channel.channelType));
        }
        return channels;
      });

  mlir_attribute_subclass(m, "AppIDAttr", circtESIAttributeIsAnAppIDAttr)
      .def_classmethod(
          "get",
          [](py::object cls, std::string name, std::optional<uint64_t> index,
             MlirContext ctxt) {
            if (index.has_value())
              return cls(circtESIAppIDAttrGet(ctxt, wrap(name), index.value()));
            return cls(circtESIAppIDAttrGetNoIdx(ctxt, wrap(name)));
          },
          "Create an AppID attribute", py::arg("cls"), py::arg("name"),
          py::arg("index") = py::none(), py::arg("context") = py::none())
      .def_property_readonly("name",
                             [](MlirAttribute self) {
                               llvm::StringRef name =
                                   unwrap(circtESIAppIDAttrGetName(self));
                               return std::string(name.data(), name.size());
                             })
      .def_property_readonly("index", [](MlirAttribute self) -> py::object {
        uint64_t index;
        if (circtESIAppIDAttrGetIndex(self, &index))
          return py::cast(index);
        return py::none();
      });

  mlir_attribute_subclass(m, "AppIDPathAttr",
                          circtESIAttributeIsAnAppIDPathAttr)
      .def_classmethod(
          "get",
          [](py::object cls, MlirAttribute root,
             std::vector<MlirAttribute> path, MlirContext ctxt) {
            return cls(
                circtESIAppIDAttrPathGet(ctxt, root, path.size(), path.data()));
          },
          "Create an AppIDPath attribute", py::arg("cls"), py::arg("root"),
          py::arg("path"), py::arg("context") = py::none())
      .def_property_readonly("root", &circtESIAppIDAttrPathGetRoot)
      .def("__len__", &circtESIAppIDAttrPathGetNumComponents)
      .def("__getitem__", &circtESIAppIDAttrPathGetComponent);

  m.def("check_inner_type_match", &circtESICheckInnerTypeMatch,
        "Check that two types match, allowing for AnyType in 'expected'.",
        py::arg("expected"), py::arg("actual"));

  py::class_<PyAppIDIndex>(m, "AppIDIndex")
      .def(py::init<MlirOperation>(), py::arg("root"))
      .def("get_child_appids_of", &PyAppIDIndex::getChildAppIDsOf,
           "Return a dictionary of AppIDAttrs to ArrayAttr of InnerRefAttrs "
           "containing the relative paths to the leaf of the particular "
           "AppIDAttr. Argument MUST be HWModuleLike.",
           py::arg("mod"))
      .def("get_appid_path", &PyAppIDIndex::getAppIDPathAttr,
           "Return an array of InnerNameRefAttrs representing the relative "
           "path to 'appid' from 'fromMod'.",
           py::arg("from_mod"), py::arg("appid"),
           py::arg("query_site") = py::none());
}
