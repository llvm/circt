//===- ESIModule.cpp - ESI API nanobind module ----------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "CIRCTModules.h"

#include "circt/Dialect/ESI/ESIDialect.h"

#include "circt-c/Dialect/ESI.h"
#include "circt-c/Dialect/HW.h"
#include "mlir-c/Bindings/Python/Interop.h"

#include "mlir/Bindings/Python/NanobindAdaptors.h"
#include "mlir/CAPI/IR.h"
#include "mlir/CAPI/Support.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"

#include "NanobindUtils.h"
#include <nanobind/nanobind.h>
namespace nb = nanobind;

using namespace circt::esi;

//===----------------------------------------------------------------------===//
// The main entry point into the ESI Assembly API.
//===----------------------------------------------------------------------===//

/// Container for a Python function that will be called to generate a service.
class ServiceGenFunc {
public:
  ServiceGenFunc(nb::object genFunc) : genFunc(std::move(genFunc)) {}

  MlirLogicalResult run(MlirOperation reqOp, MlirOperation declOp,
                        MlirOperation recOp) {
    nb::gil_scoped_acquire acquire;
    nb::object rc = genFunc(reqOp, declOp, recOp);
    return nb::cast<bool>(rc) ? mlirLogicalResultSuccess()
                              : mlirLogicalResultFailure();
  }

private:
  nb::object genFunc;
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

void registerServiceGenerator(std::string name, nb::object genFunc) {
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

  nb::object getAppIDPathAttr(MlirOperation fromMod, MlirAttribute appid,
                              MlirLocation loc) const {
    MlirAttribute path =
        circtESIAppIDIndexGetAppIDPath(index, fromMod, appid, loc);
    if (path.ptr == nullptr)
      return nb::none();
    return nb::cast(path);
  }

private:
  CirctESIAppIDIndex index;
};

using namespace mlir::python::nanobind_adaptors;

void circt::python::populateDialectESISubmodule(nb::module_ &m) {
  m.doc() = "ESI Python Native Extension";
  ::registerESIPasses();

  // Clean up references when the module is unloaded.
  auto cleanup = []() { serviceGenFuncLookup.clear(); };
  m.def("cleanup", cleanup,
        "Cleanup various references. Must be called before the module is "
        "unloaded in order to not leak.");

  m.def("registerServiceGenerator", registerServiceGenerator,
        "Register a service generator for a given service name.",
        nb::arg("impl_type"), nb::arg("generator"));

  mlir_type_subclass(m, "ChannelType", circtESITypeIsAChannelType)
      .def_classmethod(
          "get",
          [](nb::object cls, MlirType inner, uint32_t signaling = 0,
             uint64_t dataDelay = 0) {
            if (circtESITypeIsAChannelType(inner))
              return cls(inner);
            return cls(circtESIChannelTypeGet(inner, signaling, dataDelay));
          },
          nb::arg("cls"), nb::arg("inner"), nb::arg("signaling") = 0,
          nb::arg("dataDelay") = 0)
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
          [](nb::object cls, MlirContext ctxt) {
            return cls(circtESIAnyTypeGet(ctxt));
          },
          nb::arg("self"), nb::arg("ctxt") = nullptr);

  mlir_type_subclass(m, "ListType", circtESITypeIsAListType)
      .def_classmethod(
          "get",
          [](nb::object cls, MlirType inner) {
            return cls(circtESIListTypeGet(inner));
          },
          nb::arg("cls"), nb::arg("inner"))
      .def_property_readonly("element_type", [](MlirType self) {
        return circtESIListTypeGetElementType(self);
      });

  nb::enum_<ChannelDirection>(m, "ChannelDirection")
      .value("TO", ChannelDirection::to)
      .value("FROM", ChannelDirection::from);
  mlir_type_subclass(m, "BundleType", circtESITypeIsABundleType)
      .def_classmethod(
          "get",
          [](nb::object cls, std::vector<nb::tuple> channelTuples,
             bool resettable, MlirContext ctxt) {
            llvm::SmallVector<CirctESIBundleTypeBundleChannel, 4> channels(
                llvm::map_range(channelTuples, [ctxt](nb::tuple t) {
                  std::string name = nb::cast<std::string>(t[0]);
                  return CirctESIBundleTypeBundleChannel{
                      mlirIdentifierGet(ctxt, mlirStringRefCreate(
                                                  name.data(), name.length())),
                      (uint32_t)nb::cast<ChannelDirection>(t[1]),
                      nb::cast<MlirType>(t[2])};
                }));
            return cls(circtESIBundleTypeGet(ctxt, channels.size(),
                                             channels.data(), resettable));
          },
          nb::arg("cls"), nb::arg("channels"), nb::arg("resettable"),
          nb::arg("ctxt") = nullptr)
      .def_property_readonly("resettable", &circtESIBundleTypeGetResettable)
      .def_property_readonly("channels", [](MlirType bundleType) {
        std::vector<nb::tuple> channels;
        size_t numChannels = circtESIBundleTypeGetNumChannels(bundleType);
        for (size_t i = 0; i < numChannels; ++i) {
          CirctESIBundleTypeBundleChannel channel =
              circtESIBundleTypeGetChannel(bundleType, i);
          MlirStringRef name = mlirIdentifierStr(channel.name);
          channels.push_back(nb::make_tuple(nb::str(name.data, name.length),
                                            (ChannelDirection)channel.direction,
                                            channel.channelType));
        }
        return channels;
      });

  mlir_type_subclass(m, "WindowType", circtESITypeIsAWindowType)
      .def_classmethod(
          "get",
          [](nb::object cls, MlirAttribute name, MlirType into,
             std::vector<MlirType> frames, MlirContext ctxt) {
            if (!hwTypeIsAStructType(into) &&
                (!hwTypeIsATypeAliasType(into) ||
                 !hwTypeIsAStructType(hwTypeAliasTypeGetInnerType(into))))
              throw nb::type_error("'into' type must be a hw.StructType");

            // Verify all frames are WindowFrameTypes
            for (const auto &frame : frames) {
              if (!circtESITypeIsAWindowFrameType(frame)) {
                throw nb::type_error("All frames must be WindowFrameTypes");
              }
            }
            return cls(circtESIWindowTypeGet(ctxt, name, into, frames.size(),
                                             frames.data()));
          },
          nb::arg("cls"), nb::arg("name"), nb::arg("into"), nb::arg("frames"),
          nb::arg("ctxt") = nullptr)
      .def_property_readonly("name", &circtESIWindowTypeGetName)
      .def_property_readonly("into", &circtESIWindowTypeGetInto)
      .def_property_readonly(
          "frames",
          [](MlirType windowType) {
            std::vector<MlirType> frames;
            size_t numFrames = circtESIWindowTypeGetNumFrames(windowType);
            for (size_t i = 0; i < numFrames; ++i)
              frames.push_back(circtESIWindowTypeGetFrame(windowType, i));
            return frames;
          })
      .def("get_lowered_type", &circtESIWindowTypeGetLoweredType);

  mlir_type_subclass(m, "WindowFrameType", circtESITypeIsAWindowFrameType)
      .def_classmethod(
          "get",
          [](nb::object cls, MlirAttribute name, std::vector<MlirType> members,
             MlirContext ctxt) {
            // Verify all members are WindowFieldTypes
            for (const auto &member : members) {
              if (!circtESITypeIsAWindowFieldType(member)) {
                throw nb::type_error("All members must be WindowFieldTypes");
              }
            }
            return cls(circtESIWindowFrameTypeGet(ctxt, name, members.size(),
                                                  members.data()));
          },
          nb::arg("cls"), nb::arg("name"), nb::arg("members"),
          nb::arg("ctxt") = nullptr)
      .def_property_readonly("name", &circtESIWindowFrameTypeGetName)
      .def_property_readonly("members", [](MlirType frameType) {
        std::vector<MlirType> members;
        size_t numMembers = circtESIWindowFrameTypeGetNumMembers(frameType);
        for (size_t i = 0; i < numMembers; ++i)
          members.push_back(circtESIWindowFrameTypeGetMember(frameType, i));
        return members;
      });

  mlir_type_subclass(m, "WindowFieldType", circtESITypeIsAWindowFieldType)
      .def_classmethod(
          "get",
          [](nb::object cls, MlirAttribute fieldName, uint64_t numItems,
             uint64_t bulkCountWidth, MlirContext ctxt) {
            return cls(circtESIWindowFieldTypeGet(ctxt, fieldName, numItems,
                                                  bulkCountWidth));
          },
          nb::arg("cls"), nb::arg("field_name"), nb::arg("num_items") = 0,
          nb::arg("bulk_count_width") = 0, nb::arg("ctxt") = nullptr)
      .def_property_readonly("field_name", &circtESIWindowFieldTypeGetFieldName)
      .def_property_readonly("num_items", &circtESIWindowFieldTypeGetNumItems)
      .def_property_readonly("bulk_count_width",
                             &circtESIWindowFieldTypeGetBulkCountWidth);

  mlir_attribute_subclass(m, "AppIDAttr", circtESIAttributeIsAnAppIDAttr)
      .def_classmethod(
          "get",
          [](nb::object cls, std::string name, std::optional<uint64_t> index,
             MlirContext ctxt) {
            if (index.has_value())
              return cls(circtESIAppIDAttrGet(ctxt, wrap(name), index.value()));
            return cls(circtESIAppIDAttrGetNoIdx(ctxt, wrap(name)));
          },
          "Create an AppID attribute", nb::arg("cls"), nb::arg("name"),
          nb::arg("index") = nb::none(), nb::arg("context") = nb::none())
      .def_property_readonly("name",
                             [](MlirAttribute self) {
                               llvm::StringRef name =
                                   unwrap(circtESIAppIDAttrGetName(self));
                               return std::string(name.data(), name.size());
                             })
      .def_property_readonly("index", [](MlirAttribute self) -> nb::object {
        uint64_t index;
        if (circtESIAppIDAttrGetIndex(self, &index))
          return nb::cast(index);
        return nb::none();
      });

  mlir_attribute_subclass(m, "AppIDPathAttr",
                          circtESIAttributeIsAnAppIDPathAttr)
      .def_classmethod(
          "get",
          [](nb::object cls, MlirAttribute root,
             std::vector<MlirAttribute> path, MlirContext ctxt) {
            return cls(
                circtESIAppIDAttrPathGet(ctxt, root, path.size(), path.data()));
          },
          "Create an AppIDPath attribute", nb::arg("cls"), nb::arg("root"),
          nb::arg("path"), nb::arg("context") = nb::none())
      .def_property_readonly("root", &circtESIAppIDAttrPathGetRoot)
      .def("__len__", &circtESIAppIDAttrPathGetNumComponents)
      .def("__getitem__", &circtESIAppIDAttrPathGetComponent);

  m.def("check_inner_type_match", &circtESICheckInnerTypeMatch,
        "Check that two types match, allowing for AnyType in 'expected'.",
        nb::arg("expected"), nb::arg("actual"));

  nb::class_<PyAppIDIndex>(m, "AppIDIndex")
      .def(nb::init<MlirOperation>(), nb::arg("root"))
      .def("get_child_appids_of", &PyAppIDIndex::getChildAppIDsOf,
           "Return a dictionary of AppIDAttrs to ArrayAttr of InnerRefAttrs "
           "containing the relative paths to the leaf of the particular "
           "AppIDAttr. Argument MUST be HWModuleLike.",
           nb::arg("mod"))
      .def("get_appid_path", &PyAppIDIndex::getAppIDPathAttr,
           "Return an array of InnerNameRefAttrs representing the relative "
           "path to 'appid' from 'fromMod'.",
           nb::arg("from_mod"), nb::arg("appid"),
           nb::arg("query_site") = nb::none());
}
