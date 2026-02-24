//===- HWModule.cpp - HW API nanobind module ------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "CIRCTModules.h"

#include "circt-c/Dialect/HW.h"

#include "mlir-c/BuiltinAttributes.h"
#include "mlir/Bindings/Python/IRCore.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/Support/raw_ostream.h"

#include "NanobindUtils.h"
#include "mlir-c/IR.h"
#include "mlir-c/Support.h"
#include <nanobind/nanobind.h>
namespace nb = nanobind;

using namespace circt;
using mlir::python::MLIR_BINDINGS_PYTHON_DOMAIN::DefaultingPyMlirContext;
using mlir::python::MLIR_BINDINGS_PYTHON_DOMAIN::PyConcreteAttribute;
using mlir::python::MLIR_BINDINGS_PYTHON_DOMAIN::PyConcreteType;
using mlir::python::MLIR_BINDINGS_PYTHON_DOMAIN::PyMlirContext;

//===----------------------------------------------------------------------===//
// Type bindings
//===----------------------------------------------------------------------===//

struct PyInOutType : PyConcreteType<PyInOutType> {
  static constexpr IsAFunctionTy isaFunction = hwTypeIsAInOut;
  static constexpr const char *pyClassName = "InOutType";
  using Base::Base;

  static void bindDerived(ClassTy &c) {
    c.def_static("get",
                 [](MlirType innerType) {
                   auto type = hwInOutTypeGet(innerType);
                   return PyInOutType(
                       PyMlirContext::forContext(mlirTypeGetContext(type)),
                       type);
                 });
    c.def_prop_ro("element_type", [](PyInOutType &self) {
      return hwInOutTypeGetElementType(self);
    });
  }
};

struct PyHWArrayType : PyConcreteType<PyHWArrayType> {
  static constexpr IsAFunctionTy isaFunction = hwTypeIsAArrayType;
  static constexpr const char *pyClassName = "ArrayType";
  using Base::Base;

  static void bindDerived(ClassTy &c) {
    c.def_static("get",
                 [](MlirType elementType, intptr_t size) {
                   auto type = hwArrayTypeGet(elementType, size);
                   return PyHWArrayType(
                       PyMlirContext::forContext(mlirTypeGetContext(type)),
                       type);
                 });
    c.def_prop_ro("element_type", [](PyHWArrayType &self) {
      return hwArrayTypeGetElementType(self);
    });
    c.def_prop_ro(
        "size", [](PyHWArrayType &self) { return hwArrayTypeGetSize(self); });
  }
};

struct PyModuleType : PyConcreteType<PyModuleType> {
  static constexpr IsAFunctionTy isaFunction = hwTypeIsAModuleType;
  static constexpr const char *pyClassName = "ModuleType";
  using Base::Base;

  static void bindDerived(ClassTy &c) {
    c.def_static(
        "get",
        [](nb::list pyModulePorts, DefaultingPyMlirContext ctx) {
          std::vector<HWModulePort> modulePorts;
          for (auto pyModulePort : pyModulePorts)
            modulePorts.push_back(nb::cast<HWModulePort>(pyModulePort));

          return PyModuleType(
              ctx->getRef(),
              hwModuleTypeGet(ctx->get(), modulePorts.size(),
                              modulePorts.data()));
        },
        nb::arg("ports"), nb::arg("context").none() = nb::none());
    c.def_prop_ro("input_types",
                  [](PyModuleType &self) {
                    nb::list inputTypes;
                    intptr_t numInputs = hwModuleTypeGetNumInputs(self);
                    for (intptr_t i = 0; i < numInputs; ++i)
                      inputTypes.append(hwModuleTypeGetInputType(self, i));
                    return inputTypes;
                  });
    c.def_prop_ro("input_names",
                  [](PyModuleType &self) {
                    std::vector<std::string> inputNames;
                    intptr_t numInputs = hwModuleTypeGetNumInputs(self);
                    for (intptr_t i = 0; i < numInputs; ++i) {
                      auto name = hwModuleTypeGetInputName(self, i);
                      inputNames.emplace_back(name.data, name.length);
                    }
                    return inputNames;
                  });
    c.def_prop_ro("output_types",
                  [](PyModuleType &self) {
                    nb::list outputTypes;
                    intptr_t numOutputs = hwModuleTypeGetNumOutputs(self);
                    for (intptr_t i = 0; i < numOutputs; ++i)
                      outputTypes.append(hwModuleTypeGetOutputType(self, i));
                    return outputTypes;
                  });
    c.def_prop_ro("output_names", [](PyModuleType &self) {
      std::vector<std::string> outputNames;
      intptr_t numOutputs = hwModuleTypeGetNumOutputs(self);
      for (intptr_t i = 0; i < numOutputs; ++i) {
        auto name = hwModuleTypeGetOutputName(self, i);
        outputNames.emplace_back(name.data, name.length);
      }
      return outputNames;
    });
  }
};

struct PyParamIntType : PyConcreteType<PyParamIntType> {
  static constexpr IsAFunctionTy isaFunction = hwTypeIsAIntType;
  static constexpr const char *pyClassName = "ParamIntType";
  using Base::Base;

  static void bindDerived(ClassTy &c) {
    c.def_static("get_from_param",
                 [](MlirContext ctx, MlirAttribute param) {
                   auto type = hwParamIntTypeGet(param);
                   return PyParamIntType(
                       PyMlirContext::forContext(mlirTypeGetContext(type)),
                       type);
                 });
    c.def_prop_ro("width", [](PyParamIntType &self) {
      return hwParamIntTypeGetWidthAttr(self);
    });
  }
};

struct PyStructType : PyConcreteType<PyStructType> {
  static constexpr IsAFunctionTy isaFunction = hwTypeIsAStructType;
  static constexpr const char *pyClassName = "StructType";
  using Base::Base;

  static void bindDerived(ClassTy &c) {
    c.def_static(
        "get",
        [](nb::list pyFieldInfos, DefaultingPyMlirContext context) {
          llvm::SmallVector<HWStructFieldInfo> mlirFieldInfos;
          MlirContext ctx = context.resolve().get();

          llvm::SmallVector<llvm::SmallString<8>> names;
          for (size_t i = 0, e = pyFieldInfos.size(); i < e; ++i) {
            auto tuple = nb::cast<nb::tuple>(pyFieldInfos[i]);
            auto type = nb::cast<MlirType>(tuple[1]);
            if (mlirContextIsNull(ctx)) {
              ctx = mlirTypeGetContext(type);
            }
            names.emplace_back(nb::cast<std::string>(tuple[0]));
            auto nameStringRef =
                mlirStringRefCreate(names[i].data(), names[i].size());
            mlirFieldInfos.push_back(HWStructFieldInfo{
                mlirIdentifierGet(ctx, nameStringRef), type});
          }
          if (mlirContextIsNull(ctx)) {
            throw std::invalid_argument(
                "StructType requires a context if no fields provided.");
          }
          return PyStructType(
              PyMlirContext::forContext(ctx),
              hwStructTypeGet(ctx, mlirFieldInfos.size(),
                              mlirFieldInfos.data()));
        },
        nb::arg("fields"), nb::arg("context").none() = nb::none());
    c.def("get_field",
          [](PyStructType &self, std::string fieldName) {
            return hwStructTypeGetField(
                self, mlirStringRefCreateFromCString(fieldName.c_str()));
          });
    c.def("get_field_index",
          [](PyStructType &self, const std::string &fieldName) {
            return hwStructTypeGetFieldIndex(
                self, mlirStringRefCreateFromCString(fieldName.c_str()));
          });
    c.def("get_fields", [](PyStructType &self) {
      intptr_t num_fields = hwStructTypeGetNumFields(self);
      nb::list fields;
      for (intptr_t i = 0; i < num_fields; ++i) {
        auto field = hwStructTypeGetFieldNum(self, i);
        auto fieldName = mlirIdentifierStr(field.name);
        std::string name(fieldName.data, fieldName.length);
        fields.append(nb::make_tuple(name, field.type));
      }
      return fields;
    });
  }
};

struct PyUnionType : PyConcreteType<PyUnionType> {
  static constexpr IsAFunctionTy isaFunction = hwTypeIsAUnionType;
  static constexpr const char *pyClassName = "UnionType";
  using Base::Base;

  static void bindDerived(ClassTy &c) {
    c.def_static(
        "get",
        [](nb::list pyFieldInfos) {
          llvm::SmallVector<HWUnionFieldInfo> mlirFieldInfos;
          MlirContext ctx;

          llvm::SmallVector<llvm::SmallString<8>> names;
          for (size_t i = 0, e = pyFieldInfos.size(); i < e; ++i) {
            auto tuple = nb::cast<nb::tuple>(pyFieldInfos[i]);
            if (tuple.size() < 3)
              throw std::invalid_argument(
                  "UnionType field info must be a tuple of (name, type, "
                  "offset)");
            auto type = nb::cast<MlirType>(tuple[1]);
            size_t offset = nb::cast<size_t>(tuple[2]);
            ctx = mlirTypeGetContext(type);
            names.emplace_back(nb::cast<std::string>(tuple[0]));
            auto nameStringRef =
                mlirStringRefCreate(names[i].data(), names[i].size());
            mlirFieldInfos.push_back(HWUnionFieldInfo{
                mlirIdentifierGet(ctx, nameStringRef), type, offset});
          }
          return PyUnionType(
              PyMlirContext::forContext(ctx),
              hwUnionTypeGet(ctx, mlirFieldInfos.size(),
                             mlirFieldInfos.data()));
        });
    c.def("get_field",
          [](PyUnionType &self, std::string fieldName) {
            return hwUnionTypeGetField(
                self, mlirStringRefCreateFromCString(fieldName.c_str()));
          });
    c.def("get_field_index",
          [](PyUnionType &self, const std::string &fieldName) {
            return hwUnionTypeGetFieldIndex(
                self, mlirStringRefCreateFromCString(fieldName.c_str()));
          });
    c.def("get_fields", [](PyUnionType &self) {
      intptr_t num_fields = hwUnionTypeGetNumFields(self);
      nb::list fields;
      for (intptr_t i = 0; i < num_fields; ++i) {
        auto field = hwUnionTypeGetFieldNum(self, i);
        auto fieldName = mlirIdentifierStr(field.name);
        std::string name(fieldName.data, fieldName.length);
        fields.append(nb::make_tuple(name, field.type, field.offset));
      }
      return fields;
    });
  }
};

struct PyTypeAliasType : PyConcreteType<PyTypeAliasType> {
  static constexpr IsAFunctionTy isaFunction = hwTypeIsATypeAliasType;
  static constexpr const char *pyClassName = "TypeAliasType";
  using Base::Base;

  static void bindDerived(ClassTy &c) {
    c.def_static("get",
                 [](std::string scope, std::string name,
                    MlirType innerType) {
                   auto type = hwTypeAliasTypeGet(
                       mlirStringRefCreateFromCString(scope.c_str()),
                       mlirStringRefCreateFromCString(name.c_str()),
                       innerType);
                   return PyTypeAliasType(
                       PyMlirContext::forContext(mlirTypeGetContext(type)),
                       type);
                 });
    c.def_prop_ro("canonical_type", [](PyTypeAliasType &self) {
      return hwTypeAliasTypeGetCanonicalType(self);
    });
    c.def_prop_ro("inner_type", [](PyTypeAliasType &self) {
      return hwTypeAliasTypeGetInnerType(self);
    });
    c.def_prop_ro("name",
                  [](PyTypeAliasType &self) {
                    MlirStringRef cStr = hwTypeAliasTypeGetName(self);
                    return std::string(cStr.data, cStr.length);
                  });
    c.def_prop_ro("scope", [](PyTypeAliasType &self) {
      MlirStringRef cStr = hwTypeAliasTypeGetScope(self);
      return std::string(cStr.data, cStr.length);
    });
  }
};

//===----------------------------------------------------------------------===//
// Attribute bindings
//===----------------------------------------------------------------------===//

struct PyParamDeclAttr : PyConcreteAttribute<PyParamDeclAttr> {
  static constexpr IsAFunctionTy isaFunction = hwAttrIsAParamDeclAttr;
  static constexpr const char *pyClassName = "ParamDeclAttr";
  using Base::Base;

  static void bindDerived(ClassTy &c) {
    c.def_static("get",
                 [](std::string name, MlirType type, MlirAttribute value) {
                   auto attr = hwParamDeclAttrGet(
                       mlirStringRefCreateFromCString(name.c_str()), type,
                       value);
                   return PyParamDeclAttr(
                       PyMlirContext::forContext(mlirAttributeGetContext(attr)),
                       attr);
                 });
    c.def_static("get_nodefault",
                 [](std::string name, MlirType type) {
                   auto attr = hwParamDeclAttrGet(
                       mlirStringRefCreateFromCString(name.c_str()), type,
                       MlirAttribute{nullptr});
                   return PyParamDeclAttr(
                       PyMlirContext::forContext(mlirAttributeGetContext(attr)),
                       attr);
                 });
    c.def_prop_ro("value", [](PyParamDeclAttr &self) {
      return hwParamDeclAttrGetValue(self);
    });
    c.def_prop_ro("param_type", [](PyParamDeclAttr &self) {
      return hwParamDeclAttrGetType(self);
    });
    c.def_prop_ro("name", [](PyParamDeclAttr &self) {
      MlirStringRef cStr = hwParamDeclAttrGetName(self);
      return std::string(cStr.data, cStr.length);
    });
  }
};

struct PyParamDeclRefAttr : PyConcreteAttribute<PyParamDeclRefAttr> {
  static constexpr IsAFunctionTy isaFunction = hwAttrIsAParamDeclRefAttr;
  static constexpr const char *pyClassName = "ParamDeclRefAttr";
  using Base::Base;

  static void bindDerived(ClassTy &c) {
    c.def_static(
        "get",
        [](MlirContext ctx, std::string name) {
          auto attr = hwParamDeclRefAttrGet(
              ctx, mlirStringRefCreateFromCString(name.c_str()));
          return PyParamDeclRefAttr(
              PyMlirContext::forContext(mlirAttributeGetContext(attr)), attr);
        });
    c.def_prop_ro("param_type", [](PyParamDeclRefAttr &self) {
      return hwParamDeclRefAttrGetType(self);
    });
    c.def_prop_ro("name", [](PyParamDeclRefAttr &self) {
      MlirStringRef cStr = hwParamDeclRefAttrGetName(self);
      return std::string(cStr.data, cStr.length);
    });
  }
};

struct PyParamVerbatimAttr : PyConcreteAttribute<PyParamVerbatimAttr> {
  static constexpr IsAFunctionTy isaFunction = hwAttrIsAParamVerbatimAttr;
  static constexpr const char *pyClassName = "ParamVerbatimAttr";
  using Base::Base;

  static void bindDerived(ClassTy &c) {
    c.def_static("get", [](MlirAttribute text) {
      auto attr = hwParamVerbatimAttrGet(text);
      return PyParamVerbatimAttr(
          PyMlirContext::forContext(mlirAttributeGetContext(attr)), attr);
    });
  }
};

struct PyOutputFileAttr : PyConcreteAttribute<PyOutputFileAttr> {
  static constexpr IsAFunctionTy isaFunction = hwAttrIsAOutputFileAttr;
  static constexpr const char *pyClassName = "OutputFileAttr";
  using Base::Base;

  static void bindDerived(ClassTy &c) {
    c.def_static("get_from_filename",
                 [](MlirAttribute fileName, bool excludeFromFileList,
                    bool includeReplicatedOp) {
                   auto attr = hwOutputFileGetFromFileName(
                       fileName, excludeFromFileList, includeReplicatedOp);
                   return PyOutputFileAttr(
                       PyMlirContext::forContext(mlirAttributeGetContext(attr)),
                       attr);
                 });
    c.def_prop_ro("filename", [](PyOutputFileAttr &self) {
      MlirStringRef cStr = hwOutputFileGetFileName(self);
      return std::string(cStr.data, cStr.length);
    });
  }
};

struct PyInnerSymAttr : PyConcreteAttribute<PyInnerSymAttr> {
  static constexpr IsAFunctionTy isaFunction = hwAttrIsAInnerSymAttr;
  static constexpr const char *pyClassName = "InnerSymAttr";
  using Base::Base;

  static void bindDerived(ClassTy &c) {
    c.def_static("get", [](MlirAttribute symName) {
      auto attr = hwInnerSymAttrGet(symName);
      return PyInnerSymAttr(
          PyMlirContext::forContext(mlirAttributeGetContext(attr)), attr);
    });
    c.def_prop_ro("symName", [](PyInnerSymAttr &self) {
      return hwInnerSymAttrGetSymName(self);
    });
  }
};

struct PyInnerRefAttr : PyConcreteAttribute<PyInnerRefAttr> {
  static constexpr IsAFunctionTy isaFunction = hwAttrIsAInnerRefAttr;
  static constexpr const char *pyClassName = "InnerRefAttr";
  using Base::Base;

  static void bindDerived(ClassTy &c) {
    c.def_static(
        "get",
        [](MlirAttribute moduleName, MlirAttribute innerSym) {
          auto attr = hwInnerRefAttrGet(moduleName, innerSym);
          return PyInnerRefAttr(
              PyMlirContext::forContext(mlirAttributeGetContext(attr)), attr);
        });
    c.def_prop_ro("module", [](PyInnerRefAttr &self) {
      return hwInnerRefAttrGetModule(self);
    });
    c.def_prop_ro("name", [](PyInnerRefAttr &self) {
      return hwInnerRefAttrGetName(self);
    });
  }
};

/// Populate the hw python module.
void circt::python::populateDialectHWSubmodule(nb::module_ &m) {
  m.doc() = "HW dialect Python native extension";

  m.def("get_bitwidth", &hwGetBitWidth);

  PyInOutType::bind(m);
  PyHWArrayType::bind(m);

  nb::enum_<HWModulePortDirection>(m, "ModulePortDirection")
      .value("INPUT", HWModulePortDirection::Input)
      .value("OUTPUT", HWModulePortDirection::Output)
      .value("INOUT", HWModulePortDirection::InOut)
      .export_values();

  nb::class_<HWModulePort>(m, "ModulePort")
      .def(nb::init<MlirAttribute, MlirType, HWModulePortDirection>());

  PyModuleType::bind(m);
  PyParamIntType::bind(m);
  PyStructType::bind(m);
  PyUnionType::bind(m);
  PyTypeAliasType::bind(m);

  PyParamDeclAttr::bind(m);
  PyParamDeclRefAttr::bind(m);
  PyParamVerbatimAttr::bind(m);
  PyOutputFileAttr::bind(m);
  PyInnerSymAttr::bind(m);
  PyInnerRefAttr::bind(m);
}
