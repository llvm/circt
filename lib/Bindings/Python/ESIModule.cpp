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

#include "mlir/Bindings/Python/IRCore.h"
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

using mlir::python::MLIR_BINDINGS_PYTHON_DOMAIN::DefaultingPyMlirContext;
using mlir::python::MLIR_BINDINGS_PYTHON_DOMAIN::PyConcreteAttribute;
using mlir::python::MLIR_BINDINGS_PYTHON_DOMAIN::PyConcreteType;
using mlir::python::MLIR_BINDINGS_PYTHON_DOMAIN::PyMlirContext;

//===----------------------------------------------------------------------===//
// Type bindings
//===----------------------------------------------------------------------===//

struct PyChannelType : PyConcreteType<PyChannelType> {
  static constexpr IsAFunctionTy isaFunction = circtESITypeIsAChannelType;
  static constexpr GetTypeIDFunctionTy getTypeIdFunction =
      circtESIChannelTypeGetTypeID;
  static constexpr const char *pyClassName = "ChannelType";
  using Base::Base;

  static void bindDerived(ClassTy &c) {
    c.def_static(
        "get",
        [](MlirType inner, uint32_t signaling, uint64_t dataDelay) {
          MlirType type;
          if (circtESITypeIsAChannelType(inner))
            type = inner;
          else
            type = circtESIChannelTypeGet(inner, signaling, dataDelay);
          return PyChannelType(
              PyMlirContext::forContext(mlirTypeGetContext(type)), type);
        },
        nb::arg("inner"), nb::arg("signaling") = 0, nb::arg("dataDelay") = 0);
    c.def_prop_ro("inner", [](PyChannelType &self) {
      return circtESIChannelGetInner(self);
    });
    c.def_prop_ro("signaling", [](PyChannelType &self) {
      return circtESIChannelGetSignaling(self);
    });
    c.def_prop_ro("data_delay", [](PyChannelType &self) {
      return circtESIChannelGetDataDelay(self);
    });
  }
};

struct PyESIAnyType : PyConcreteType<PyESIAnyType> {
  static constexpr IsAFunctionTy isaFunction = circtESITypeIsAnAnyType;
  static constexpr GetTypeIDFunctionTy getTypeIdFunction =
      circtESIAnyTypeGetTypeID;
  static constexpr const char *pyClassName = "AnyType";
  using Base::Base;

  static void bindDerived(ClassTy &c) {
    c.def_static(
        "get",
        [](DefaultingPyMlirContext ctx) {
          return PyESIAnyType(ctx->getRef(), circtESIAnyTypeGet(ctx->get()));
        },
        nb::arg("ctxt").none() = nb::none());
  }
};

struct PyESIListType : PyConcreteType<PyESIListType> {
  static constexpr IsAFunctionTy isaFunction = circtESITypeIsAListType;
  static constexpr GetTypeIDFunctionTy getTypeIdFunction =
      circtESIListTypeGetTypeID;
  static constexpr const char *pyClassName = "ListType";
  using Base::Base;

  static void bindDerived(ClassTy &c) {
    c.def_static(
        "get",
        [](MlirType inner) {
          auto type = circtESIListTypeGet(inner);
          return PyESIListType(
              PyMlirContext::forContext(mlirTypeGetContext(type)), type);
        },
        nb::arg("inner"));
    c.def_prop_ro("element_type", [](PyESIListType &self) {
      return circtESIListTypeGetElementType(self);
    });
  }
};

struct PyBundleType : PyConcreteType<PyBundleType> {
  static constexpr IsAFunctionTy isaFunction = circtESITypeIsABundleType;
  static constexpr GetTypeIDFunctionTy getTypeIdFunction =
      circtESIBundleTypeGetTypeID;
  static constexpr const char *pyClassName = "BundleType";
  using Base::Base;

  static void bindDerived(ClassTy &c) {
    c.def_static(
        "get",
        [](std::vector<nb::tuple> channelTuples, bool resettable,
           DefaultingPyMlirContext ctx) {
          MlirContext ctxt = ctx->get();
          llvm::SmallVector<CirctESIBundleTypeBundleChannel, 4> channels(
              llvm::map_range(channelTuples, [ctxt](nb::tuple t) {
                std::string name = nb::cast<std::string>(t[0]);
                return CirctESIBundleTypeBundleChannel{
                    mlirIdentifierGet(
                        ctxt, mlirStringRefCreate(name.data(), name.length())),
                    (uint32_t)nb::cast<ChannelDirection>(t[1]),
                    nb::cast<MlirType>(t[2])};
              }));
          return PyBundleType(ctx->getRef(), circtESIBundleTypeGet(
                                                 ctxt, channels.size(),
                                                 channels.data(), resettable));
        },
        nb::arg("channels"), nb::arg("resettable"),
        nb::arg("ctxt").none() = nb::none());
    c.def_prop_ro("resettable", &circtESIBundleTypeGetResettable);
    c.def_prop_ro("channels", [](PyBundleType &self) {
      MlirType bundleType = self;
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
  }
};

struct PyWindowType : PyConcreteType<PyWindowType> {
  static constexpr IsAFunctionTy isaFunction = circtESITypeIsAWindowType;
  static constexpr GetTypeIDFunctionTy getTypeIdFunction =
      circtESIWindowTypeGetTypeID;
  static constexpr const char *pyClassName = "WindowType";
  using Base::Base;

  static void bindDerived(ClassTy &c) {
    c.def_static(
        "get",
        [](MlirAttribute name, MlirType into, std::vector<MlirType> frames,
           DefaultingPyMlirContext ctx) {
          if (!hwTypeIsAStructType(into) &&
              (!hwTypeIsATypeAliasType(into) ||
               !hwTypeIsAStructType(hwTypeAliasTypeGetInnerType(into))))
            throw nb::type_error("'into' type must be a hw.StructType");

          for (const auto &frame : frames) {
            if (!circtESITypeIsAWindowFrameType(frame)) {
              throw nb::type_error("All frames must be WindowFrameTypes");
            }
          }
          return PyWindowType(ctx->getRef(), circtESIWindowTypeGet(
                                                 ctx->get(), name, into,
                                                 frames.size(), frames.data()));
        },
        nb::arg("name"), nb::arg("into"), nb::arg("frames"),
        nb::arg("ctxt").none() = nb::none());
    c.def_prop_ro("name", &circtESIWindowTypeGetName);
    c.def_prop_ro("into", &circtESIWindowTypeGetInto);
    c.def_prop_ro("frames", [](PyWindowType &self) {
      MlirType windowType = self;
      std::vector<MlirType> frames;
      size_t numFrames = circtESIWindowTypeGetNumFrames(windowType);
      for (size_t i = 0; i < numFrames; ++i)
        frames.push_back(circtESIWindowTypeGetFrame(windowType, i));
      return frames;
    });
    c.def("get_lowered_type", &circtESIWindowTypeGetLoweredType);
  }
};

struct PyWindowFrameType : PyConcreteType<PyWindowFrameType> {
  static constexpr IsAFunctionTy isaFunction = circtESITypeIsAWindowFrameType;
  static constexpr GetTypeIDFunctionTy getTypeIdFunction =
      circtESIWindowFrameTypeGetTypeID;
  static constexpr const char *pyClassName = "WindowFrameType";
  using Base::Base;

  static void bindDerived(ClassTy &c) {
    c.def_static(
        "get",
        [](MlirAttribute name, std::vector<MlirType> members,
           DefaultingPyMlirContext ctx) {
          for (const auto &member : members) {
            if (!circtESITypeIsAWindowFieldType(member)) {
              throw nb::type_error("All members must be WindowFieldTypes");
            }
          }
          return PyWindowFrameType(ctx->getRef(),
                                   circtESIWindowFrameTypeGet(ctx->get(), name,
                                                              members.size(),
                                                              members.data()));
        },
        nb::arg("name"), nb::arg("members"),
        nb::arg("ctxt").none() = nb::none());
    c.def_prop_ro("name", &circtESIWindowFrameTypeGetName);
    c.def_prop_ro("members", [](PyWindowFrameType &self) {
      MlirType frameType = self;
      std::vector<MlirType> members;
      size_t numMembers = circtESIWindowFrameTypeGetNumMembers(frameType);
      for (size_t i = 0; i < numMembers; ++i)
        members.push_back(circtESIWindowFrameTypeGetMember(frameType, i));
      return members;
    });
  }
};

struct PyWindowFieldType : PyConcreteType<PyWindowFieldType> {
  static constexpr IsAFunctionTy isaFunction = circtESITypeIsAWindowFieldType;
  static constexpr GetTypeIDFunctionTy getTypeIdFunction =
      circtESIWindowFieldTypeGetTypeID;
  static constexpr const char *pyClassName = "WindowFieldType";
  using Base::Base;

  static void bindDerived(ClassTy &c) {
    c.def_static(
        "get",
        [](MlirAttribute fieldName, uint64_t numItems, uint64_t bulkCountWidth,
           DefaultingPyMlirContext ctx) {
          return PyWindowFieldType(
              ctx->getRef(),
              circtESIWindowFieldTypeGet(ctx->get(), fieldName, numItems,
                                         bulkCountWidth));
        },
        nb::arg("field_name"), nb::arg("num_items") = 0,
        nb::arg("bulk_count_width") = 0, nb::arg("ctxt").none() = nb::none());
    c.def_prop_ro("field_name", &circtESIWindowFieldTypeGetFieldName);
    c.def_prop_ro("num_items", &circtESIWindowFieldTypeGetNumItems);
    c.def_prop_ro("bulk_count_width",
                  &circtESIWindowFieldTypeGetBulkCountWidth);
  }
};

//===----------------------------------------------------------------------===//
// Attribute bindings
//===----------------------------------------------------------------------===//

struct PyAppIDAttr : PyConcreteAttribute<PyAppIDAttr> {
  static constexpr IsAFunctionTy isaFunction = circtESIAttributeIsAnAppIDAttr;
  static constexpr const char *pyClassName = "AppIDAttr";
  using Base::Base;

  static void bindDerived(ClassTy &c) {
    c.def_static(
        "get",
        [](std::string name, std::optional<uint64_t> index,
           DefaultingPyMlirContext ctx) {
          MlirAttribute attr;
          if (index.has_value())
            attr = circtESIAppIDAttrGet(ctx->get(), wrap(name), index.value());
          else
            attr = circtESIAppIDAttrGetNoIdx(ctx->get(), wrap(name));
          return PyAppIDAttr(ctx->getRef(), attr);
        },
        "Create an AppID attribute", nb::arg("name"),
        nb::arg("index") = nb::none(), nb::arg("context").none() = nb::none());
    c.def_prop_ro("name", [](PyAppIDAttr &self) {
      llvm::StringRef name = unwrap(circtESIAppIDAttrGetName(self));
      return std::string(name.data(), name.size());
    });
    c.def_prop_ro("index", [](PyAppIDAttr &self) -> nb::object {
      uint64_t index;
      if (circtESIAppIDAttrGetIndex(self, &index))
        return nb::cast(index);
      return nb::none();
    });
  }
};

struct PyAppIDPathAttr : PyConcreteAttribute<PyAppIDPathAttr> {
  static constexpr IsAFunctionTy isaFunction =
      circtESIAttributeIsAnAppIDPathAttr;
  static constexpr const char *pyClassName = "AppIDPathAttr";
  using Base::Base;

  static void bindDerived(ClassTy &c) {
    c.def_static(
        "get",
        [](MlirAttribute root, std::vector<MlirAttribute> path,
           DefaultingPyMlirContext ctx) {
          return PyAppIDPathAttr(
              ctx->getRef(), circtESIAppIDAttrPathGet(
                                 ctx->get(), root, path.size(), path.data()));
        },
        "Create an AppIDPath attribute", nb::arg("root"), nb::arg("path"),
        nb::arg("context").none() = nb::none());
    c.def_prop_ro("root", &circtESIAppIDAttrPathGetRoot);
    c.def("__len__", &circtESIAppIDAttrPathGetNumComponents);
    c.def("__getitem__", &circtESIAppIDAttrPathGetComponent);
  }
};

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

  PyChannelType::bind(m);
  PyESIAnyType::bind(m);
  PyESIListType::bind(m);

  nb::enum_<ChannelDirection>(m, "ChannelDirection")
      .value("TO", ChannelDirection::to)
      .value("FROM", ChannelDirection::from);

  PyBundleType::bind(m);
  PyWindowType::bind(m);
  PyWindowFrameType::bind(m);
  PyWindowFieldType::bind(m);
  PyAppIDAttr::bind(m);
  PyAppIDPathAttr::bind(m);

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
