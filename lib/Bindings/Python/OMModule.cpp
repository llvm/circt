//===- OMModule.cpp - OM API nanobind module ------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "CIRCTModules.h"
#include "circt-c/Dialect/HW.h"
#include "circt-c/Dialect/OM.h"
#include "mlir-c/BuiltinAttributes.h"
#include "mlir-c/BuiltinTypes.h"
#include "mlir-c/IR.h"
#include "mlir/Bindings/Python/IRCore.h"
#include <nanobind/nanobind.h>
#include <nanobind/stl/variant.h>
#include <nanobind/stl/vector.h>
namespace nb = nanobind;

using namespace mlir;
using namespace mlir::python;
using ::mlir::python::MLIR_BINDINGS_PYTHON_DOMAIN::PyConcreteAttribute;
using ::mlir::python::MLIR_BINDINGS_PYTHON_DOMAIN::PyConcreteType;
using ::mlir::python::MLIR_BINDINGS_PYTHON_DOMAIN::PyMlirContext;

namespace {

struct List;
struct Object;
struct BasePath;
struct Path;

using PythonPrimitive = std::variant<nb::int_, nb::float_, nb::str, nb::bool_,
                                     nb::tuple, nb::list, nb::dict>;

struct None {};
using PythonValue =
    std::variant<None, Object, List, BasePath, Path, PythonPrimitive>;

PythonValue omEvaluatorValueToPythonValue(OMEvaluatorValue result);
OMEvaluatorValue pythonValueToOMEvaluatorValue(PythonValue result,
                                               MlirContext ctx);
static PythonPrimitive omPrimitiveToPythonValue(MlirAttribute attr);
static MlirAttribute omPythonValueToPrimitive(PythonPrimitive value,
                                              MlirContext ctx);

struct List {
  List(OMEvaluatorValue value) : value(value) {}

  intptr_t getNumElements() { return omEvaluatorListGetNumElements(value); }

  PythonValue getElement(intptr_t i);
  OMEvaluatorValue getValue() const { return value; }

private:
  OMEvaluatorValue value;
};

struct BasePath {
  BasePath(OMEvaluatorValue value) : value(value) {}

  static BasePath getEmpty(MlirContext context) {
    return BasePath(omEvaluatorBasePathGetEmpty(context));
  }

  MlirContext getContext() const { return omEvaluatorValueGetContext(value); }

  OMEvaluatorValue getValue() const { return value; }

private:
  OMEvaluatorValue value;
};

struct Path {
  Path(OMEvaluatorValue value) : value(value) {}

  MlirContext getContext() const { return omEvaluatorValueGetContext(value); }

  OMEvaluatorValue getValue() const { return value; }

  std::string dunderStr() {
    auto ref = mlirStringAttrGetValue(omEvaluatorPathGetAsString(getValue()));
    return std::string(ref.data, ref.length);
  }

private:
  OMEvaluatorValue value;
};

struct Object {
  Object(OMEvaluatorValue value) : value(value) {}

  MlirType getType() { return omEvaluatorObjectGetType(value); }

  MlirLocation getLocation() { return omEvaluatorValueGetLoc(value); }

  MlirLocation getFieldLoc(const std::string &name) {
    MlirContext context = mlirTypeGetContext(omEvaluatorObjectGetType(value));
    MlirStringRef cName = mlirStringRefCreateFromCString(name.c_str());
    MlirAttribute nameAttr = mlirStringAttrGet(context, cName);

    OMEvaluatorValue result = omEvaluatorObjectGetField(value, nameAttr);

    return omEvaluatorValueGetLoc(result);
  }

  PythonValue getField(const std::string &name) {
    MlirContext context = mlirTypeGetContext(omEvaluatorObjectGetType(value));
    MlirStringRef cName = mlirStringRefCreateFromCString(name.c_str());
    MlirAttribute nameAttr = mlirStringAttrGet(context, cName);

    OMEvaluatorValue result = omEvaluatorObjectGetField(value, nameAttr);

    return omEvaluatorValueToPythonValue(result);
  }

  std::vector<std::string> getFieldNames() {
    MlirAttribute fieldNames = omEvaluatorObjectGetFieldNames(value);
    intptr_t numFieldNames = mlirArrayAttrGetNumElements(fieldNames);

    std::vector<std::string> pyFieldNames;
    for (intptr_t i = 0; i < numFieldNames; ++i) {
      MlirAttribute fieldName = mlirArrayAttrGetElement(fieldNames, i);
      MlirStringRef fieldNameStr = mlirStringAttrGetValue(fieldName);
      pyFieldNames.emplace_back(fieldNameStr.data, fieldNameStr.length);
    }

    return pyFieldNames;
  }

  unsigned getHash() { return omEvaluatorObjectGetHash(value); }

  bool eq(Object &other) { return omEvaluatorObjectIsEq(value, other.value); }

  OMEvaluatorValue getValue() const { return value; }

private:
  OMEvaluatorValue value;
};

struct Evaluator {
  Evaluator(MlirModule mod) : evaluator(omEvaluatorNew(mod)) {}

  Object instantiate(MlirAttribute className,
                     std::vector<PythonValue> actualParams) {
    std::vector<OMEvaluatorValue> values;
    for (auto &param : actualParams)
      values.push_back(pythonValueToOMEvaluatorValue(
          param, mlirModuleGetContext(getModule())));

    OMEvaluatorValue result = omEvaluatorInstantiate(
        evaluator, className, values.size(), values.data());

    if (omEvaluatorObjectIsNull(result))
      throw nb::value_error(
          "unable to instantiate object, see previous error(s)");

    return Object(result);
  }

  MlirModule getModule() { return omEvaluatorGetModule(evaluator); }

private:
  OMEvaluator evaluator;
};

class PyListAttrIterator {
public:
  PyListAttrIterator(MlirAttribute attr) : attr(std::move(attr)) {}

  PyListAttrIterator &dunderIter() { return *this; }

  MlirAttribute dunderNext() {
    if (nextIndex >= omListAttrGetNumElements(attr))
      throw nb::stop_iteration();
    return omListAttrGetElement(attr, nextIndex++);
  }

  static void bind(nb::module_ &m) {
    nb::class_<PyListAttrIterator>(m, "ListAttributeIterator")
        .def("__iter__", &PyListAttrIterator::dunderIter)
        .def("__next__", &PyListAttrIterator::dunderNext);
  }

private:
  MlirAttribute attr;
  intptr_t nextIndex = 0;
};

PythonValue List::getElement(intptr_t i) {
  return omEvaluatorValueToPythonValue(omEvaluatorListGetElement(value, i));
}

static PythonPrimitive omPrimitiveToPythonValue(MlirAttribute attr) {
  if (omAttrIsAIntegerAttr(attr)) {
    auto strRef = omIntegerAttrToString(attr);
    return nb::int_(nb::str(strRef.data, strRef.length));
  }

  if (mlirAttributeIsAFloat(attr)) {
    return nb::float_(mlirFloatAttrGetValueDouble(attr));
  }

  if (mlirAttributeIsAString(attr)) {
    auto strRef = mlirStringAttrGetValue(attr);
    return nb::str(strRef.data, strRef.length);
  }

  if (mlirAttributeIsABool(attr)) {
    return nb::bool_(mlirBoolAttrGetValue(attr));
  }

  if (mlirAttributeIsAInteger(attr)) {
    MlirType type = mlirAttributeGetType(attr);
    if (mlirTypeIsAIndex(type) || mlirIntegerTypeIsSignless(type))
      return nb::int_(mlirIntegerAttrGetValueInt(attr));
    if (mlirIntegerTypeIsSigned(type))
      return nb::int_(mlirIntegerAttrGetValueSInt(attr));
    return nb::int_(mlirIntegerAttrGetValueUInt(attr));
  }

  if (omAttrIsAReferenceAttr(attr)) {
    auto innerRef = omReferenceAttrGetInnerRef(attr);
    auto moduleStrRef =
        mlirStringAttrGetValue(hwInnerRefAttrGetModule(innerRef));
    auto nameStrRef = mlirStringAttrGetValue(hwInnerRefAttrGetName(innerRef));
    auto moduleStr = nb::str(moduleStrRef.data, moduleStrRef.length);
    auto nameStr = nb::str(nameStrRef.data, nameStrRef.length);
    return nb::make_tuple(moduleStr, nameStr);
  }

  if (omAttrIsAListAttr(attr)) {
    nb::list results;
    for (intptr_t i = 0, e = omListAttrGetNumElements(attr); i < e; ++i)
      results.append(omPrimitiveToPythonValue(omListAttrGetElement(attr, i)));
    return results;
  }

  mlirAttributeDump(attr);
  throw nb::type_error("Unexpected OM primitive attribute");
}

static MlirAttribute omPythonValueToPrimitive(PythonPrimitive value,
                                              MlirContext ctx) {
  if (auto *intValue = std::get_if<nb::int_>(&value)) {
    auto intType = mlirIntegerTypeSignedGet(ctx, 64);
    auto intAttr = mlirIntegerAttrGet(intType, nb::cast<int64_t>(*intValue));
    return omIntegerAttrGet(intAttr);
  }

  if (auto *attr = std::get_if<nb::float_>(&value)) {
    auto floatType = mlirF64TypeGet(ctx);
    return mlirFloatAttrDoubleGet(ctx, floatType, nb::cast<double>(*attr));
  }

  if (auto *attr = std::get_if<nb::str>(&value)) {
    auto str = nb::cast<std::string>(*attr);
    auto strRef = mlirStringRefCreate(str.data(), str.length());
    auto omStringType = omStringTypeGet(ctx);
    return mlirStringAttrTypedGet(omStringType, strRef);
  }

  if (auto *attr = std::get_if<nb::bool_>(&value)) {
    return mlirBoolAttrGet(ctx, nb::cast<bool>(*attr));
  }

  if (auto *attr = std::get_if<nb::list>(&value)) {
    if (attr->size() == 0)
      throw nb::type_error("Empty list is prohibited now");

    std::vector<MlirAttribute> attrs;
    attrs.reserve(attr->size());
    std::optional<MlirType> elemenType;
    for (auto v : *attr) {
      attrs.push_back(
          omPythonValueToPrimitive(nb::cast<PythonPrimitive>(v), ctx));
      if (!elemenType)
        elemenType = mlirAttributeGetType(attrs.back());
      else if (!mlirTypeEqual(*elemenType,
                              mlirAttributeGetType(attrs.back()))) {
        throw nb::type_error("List elements must be of the same type");
      }
    }
    return omListAttrGet(*elemenType, attrs.size(), attrs.data());
  }

  throw nb::type_error("Unexpected OM primitive value");
}

PythonValue omEvaluatorValueToPythonValue(OMEvaluatorValue result) {
  if (omEvaluatorValueIsNull(result))
    throw nb::value_error("unable to get field, see previous error(s)");

  if (omEvaluatorValueIsAObject(result))
    return Object(result);

  if (omEvaluatorValueIsAList(result))
    return List(result);

  if (omEvaluatorValueIsABasePath(result))
    return BasePath(result);

  if (omEvaluatorValueIsAPath(result))
    return Path(result);

  if (omEvaluatorValueIsAReference(result))
    return omEvaluatorValueToPythonValue(
        omEvaluatorValueGetReferenceValue(result));

  assert(omEvaluatorValueIsAPrimitive(result));
  return omPrimitiveToPythonValue(omEvaluatorValueGetPrimitive(result));
}

OMEvaluatorValue pythonValueToOMEvaluatorValue(PythonValue result,
                                               MlirContext ctx) {
  if (auto *list = std::get_if<List>(&result))
    return list->getValue();

  if (auto *basePath = std::get_if<BasePath>(&result))
    return basePath->getValue();

  if (auto *path = std::get_if<Path>(&result))
    return path->getValue();

  if (auto *object = std::get_if<Object>(&result))
    return object->getValue();

  auto primitive = std::get<PythonPrimitive>(result);
  return omEvaluatorValueFromPrimitive(
      omPythonValueToPrimitive(primitive, ctx));
}

} // namespace

//===----------------------------------------------------------------------===//
// Attribute types
//===----------------------------------------------------------------------===//

struct PyReferenceAttr : PyConcreteAttribute<PyReferenceAttr> {
  static constexpr IsAFunctionTy isaFunction = omAttrIsAReferenceAttr;
  static constexpr const char *pyClassName = "ReferenceAttr";
  using Base::Base;

  static void bindDerived(ClassTy &c) {
    c.def_prop_ro("inner_ref", [](PyReferenceAttr &self) {
      return omReferenceAttrGetInnerRef(self);
    });
  }
};

struct PyOMIntegerAttr : PyConcreteAttribute<PyOMIntegerAttr> {
  static constexpr IsAFunctionTy isaFunction = omAttrIsAIntegerAttr;
  static constexpr const char *pyClassName = "OMIntegerAttr";
  using Base::Base;

  static void bindDerived(ClassTy &c) {
    c.def_static("get", [](MlirAttribute intVal) {
      auto attr = omIntegerAttrGet(intVal);
      return PyOMIntegerAttr(
          PyMlirContext::forContext(mlirAttributeGetContext(attr)), attr);
    });
    c.def_prop_ro("integer", [](PyOMIntegerAttr &self) {
      return omIntegerAttrGetInt(self);
    });
    c.def("__str__", [](PyOMIntegerAttr &self) {
      MlirStringRef str = omIntegerAttrToString(self);
      return std::string(str.data, str.length);
    });
  }
};

struct PyListAttr : PyConcreteAttribute<PyListAttr> {
  static constexpr IsAFunctionTy isaFunction = omAttrIsAListAttr;
  static constexpr const char *pyClassName = "ListAttr";
  using Base::Base;

  static void bindDerived(ClassTy &c) {
    c.def("__getitem__", &omListAttrGetElement);
    c.def("__len__", &omListAttrGetNumElements);
    c.def("__iter__",
          [](MlirAttribute arr) { return PyListAttrIterator(arr); });
  }
};

//===----------------------------------------------------------------------===//
// Type types
//===----------------------------------------------------------------------===//

struct PyOMAnyType : PyConcreteType<PyOMAnyType> {
  static constexpr IsAFunctionTy isaFunction = omTypeIsAAnyType;
  static constexpr GetTypeIDFunctionTy getTypeIdFunction = omAnyTypeGetTypeID;
  static constexpr const char *pyClassName = "AnyType";
  using Base::Base;
};

struct PyClassType : PyConcreteType<PyClassType> {
  static constexpr IsAFunctionTy isaFunction = omTypeIsAClassType;
  static constexpr GetTypeIDFunctionTy getTypeIdFunction = omClassTypeGetTypeID;
  static constexpr const char *pyClassName = "ClassType";
  using Base::Base;

  static void bindDerived(ClassTy &c) {
    c.def_prop_ro("name", [](PyClassType &self) {
      MlirStringRef name = mlirIdentifierStr(omClassTypeGetName(self));
      return std::string(name.data, name.length);
    });
  }
};

struct PyBasePathType : PyConcreteType<PyBasePathType> {
  static constexpr IsAFunctionTy isaFunction = omTypeIsAFrozenBasePathType;
  static constexpr GetTypeIDFunctionTy getTypeIdFunction =
      omFrozenBasePathTypeGetTypeID;
  static constexpr const char *pyClassName = "BasePathType";
  using Base::Base;
};

struct PyOMListType : PyConcreteType<PyOMListType> {
  static constexpr IsAFunctionTy isaFunction = omTypeIsAListType;
  static constexpr GetTypeIDFunctionTy getTypeIdFunction = omListTypeGetTypeID;
  static constexpr const char *pyClassName = "ListType";
  using Base::Base;

  static void bindDerived(ClassTy &c) {
    c.def_prop_ro("element_type", omListTypeGetElementType);
  }
};

struct PyPathType : PyConcreteType<PyPathType> {
  static constexpr IsAFunctionTy isaFunction = omTypeIsAFrozenPathType;
  static constexpr GetTypeIDFunctionTy getTypeIdFunction =
      omFrozenPathTypeGetTypeID;
  static constexpr const char *pyClassName = "PathType";
  using Base::Base;
};

/// Populate the OM Python module.
void circt::python::populateDialectOMSubmodule(nb::module_ &m) {
  m.doc() = "OM dialect Python native extension";

  nb::class_<Evaluator>(m, "Evaluator")
      .def(nb::init<MlirModule>(), nb::arg("module"))
      .def("instantiate", &Evaluator::instantiate, "Instantiate an Object",
           nb::arg("class_name"), nb::arg("actual_params"))
      .def_prop_ro("module", &Evaluator::getModule,
                   "The Module the Evaluator is built from");

  nb::class_<List>(m, "List")
      .def(nb::init<List>(), nb::arg("list"))
      .def("__getitem__", &List::getElement)
      .def("__len__", &List::getNumElements);

  nb::class_<BasePath>(m, "BasePath")
      .def(nb::init<BasePath>(), nb::arg("basepath"))
      .def_static("get_empty", &BasePath::getEmpty,
                  nb::arg("context") = nb::none());

  nb::class_<Path>(m, "Path")
      .def(nb::init<Path>(), nb::arg("path"))
      .def("__str__", &Path::dunderStr);

  nb::class_<Object>(m, "Object")
      .def(nb::init<Object>(), nb::arg("object"))
      .def("__getattr__", &Object::getField, "Get a field from an Object",
           nb::arg("name"))
      .def("get_field_loc", &Object::getFieldLoc,
           "Get the location of a field from an Object", nb::arg("name"))
      .def_prop_ro("field_names", &Object::getFieldNames,
                   "Get field names from an Object")
      .def_prop_ro("type", &Object::getType, "The Type of the Object")
      .def_prop_ro("loc", &Object::getLocation, "The Location of the Object")
      .def("__hash__", &Object::getHash, "Get object hash")
      .def("__eq__", &Object::eq, "Check if two objects are same");

  PyReferenceAttr::bind(m);
  PyOMIntegerAttr::bind(m);
  PyListAttr::bind(m);
  PyListAttrIterator::bind(m);

  PyOMAnyType::bind(m);
  PyClassType::bind(m);
  PyBasePathType::bind(m);
  PyOMListType::bind(m);
  PyPathType::bind(m);
}
