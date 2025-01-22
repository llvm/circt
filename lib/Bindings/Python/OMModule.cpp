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
#include "mlir/Bindings/Python/NanobindAdaptors.h"
#include <nanobind/nanobind.h>
#include <nanobind/stl/variant.h>
#include <nanobind/stl/vector.h>
namespace nb = nanobind;

using namespace mlir;
using namespace mlir::python;
using namespace mlir::python::nanobind_adaptors;

namespace {

struct List;
struct Object;
struct Tuple;
struct Map;
struct BasePath;
struct Path;

/// These are the Python types that are represented by the different primitive
/// OMEvaluatorValues as Attributes.
using PythonPrimitive = std::variant<nb::int_, nb::float_, nb::str, nb::bool_,
                                     nb::tuple, nb::list, nb::dict>;

/// None is used to by nanobind when default initializing a PythonValue. The
/// order of types in the variant matters here, and we want nanobind to try
/// casting to the Python classes defined in this file first, before
/// MlirAttribute and the upstream MLIR type casters.  If the MlirAttribute
/// is tried first, then we can hit an assert inside the MLIR codebase.
struct None {};
using PythonValue = std::variant<None, Object, List, Tuple, Map, BasePath, Path,
                                 PythonPrimitive>;

/// Map an opaque OMEvaluatorValue into a python value.
PythonValue omEvaluatorValueToPythonValue(OMEvaluatorValue result);
OMEvaluatorValue pythonValueToOMEvaluatorValue(PythonValue result,
                                               MlirContext ctx);
static PythonPrimitive omPrimitiveToPythonValue(MlirAttribute attr);
static MlirAttribute omPythonValueToPrimitive(PythonPrimitive value,
                                              MlirContext ctx);

/// Provides a List class by simply wrapping the OMObject CAPI.
struct List {
  // Instantiate a List with a reference to the underlying OMEvaluatorValue.
  List(OMEvaluatorValue value) : value(value) {}

  /// Return the number of elements.
  intptr_t getNumElements() { return omEvaluatorListGetNumElements(value); }

  PythonValue getElement(intptr_t i);
  OMEvaluatorValue getValue() const { return value; }

private:
  // The underlying CAPI value.
  OMEvaluatorValue value;
};

struct Tuple {
  // Instantiate a Tuple with a reference to the underlying OMEvaluatorValue.
  Tuple(OMEvaluatorValue value) : value(value) {}

  /// Return the number of elements.
  intptr_t getNumElements() { return omEvaluatorTupleGetNumElements(value); }

  PythonValue getElement(intptr_t i);
  OMEvaluatorValue getValue() const { return value; }

private:
  // The underlying CAPI value.
  OMEvaluatorValue value;
};

/// Provides a Map class by simply wrapping the OMObject CAPI.
struct Map {
  // Instantiate a Map with a reference to the underlying OMEvaluatorValue.
  Map(OMEvaluatorValue value) : value(value) {}

  /// Return the keys.
  std::vector<nb::str> getKeys() {
    auto attr = omEvaluatorMapGetKeys(value);
    intptr_t numFieldNames = mlirArrayAttrGetNumElements(attr);

    std::vector<nb::str> pyFieldNames;
    for (intptr_t i = 0; i < numFieldNames; ++i) {
      auto name = mlirStringAttrGetValue(mlirArrayAttrGetElement(attr, i));
      pyFieldNames.emplace_back(nb::str(name.data, name.length));
    }

    return pyFieldNames;
  }

  /// Look up the value. A key is an integer, string or attribute.
  PythonValue dunderGetItemAttr(MlirAttribute key);
  PythonValue dunderGetItemNamed(const std::string &key);
  PythonValue dunderGetItemIndexed(intptr_t key);
  PythonValue
  dunderGetItem(std::variant<intptr_t, std::string, MlirAttribute> key);

  /// Return a context from an underlying value.
  MlirContext getContext() const { return omEvaluatorValueGetContext(value); }

  OMEvaluatorValue getValue() const { return value; }
  MlirType getType() { return omEvaluatorMapGetType(value); }

private:
  // The underlying CAPI value.
  OMEvaluatorValue value;
};

/// Provides a BasePath class by simply wrapping the OMObject CAPI.
struct BasePath {
  /// Instantiate a BasePath with a reference to the underlying
  /// OMEvaluatorValue.
  BasePath(OMEvaluatorValue value) : value(value) {}

  static BasePath getEmpty(MlirContext context) {
    return BasePath(omEvaluatorBasePathGetEmpty(context));
  }

  /// Return a context from an underlying value.
  MlirContext getContext() const { return omEvaluatorValueGetContext(value); }

  OMEvaluatorValue getValue() const { return value; }

private:
  // The underlying CAPI value.
  OMEvaluatorValue value;
};

/// Provides a Path class by simply wrapping the OMObject CAPI.
struct Path {
  /// Instantiate a Path with a reference to the underlying OMEvaluatorValue.
  Path(OMEvaluatorValue value) : value(value) {}

  /// Return a context from an underlying value.
  MlirContext getContext() const { return omEvaluatorValueGetContext(value); }

  OMEvaluatorValue getValue() const { return value; }

  std::string dunderStr() {
    auto ref = mlirStringAttrGetValue(omEvaluatorPathGetAsString(getValue()));
    return std::string(ref.data, ref.length);
  }

private:
  // The underlying CAPI value.
  OMEvaluatorValue value;
};

/// Provides an Object class by simply wrapping the OMObject CAPI.
struct Object {
  // Instantiate an Object with a reference to the underlying OMObject.
  Object(OMEvaluatorValue value) : value(value) {}

  /// Get the Type from an Object, which will be a ClassType.
  MlirType getType() { return omEvaluatorObjectGetType(value); }

  /// Get the Location from an Object, which will be an MlirLocation.
  MlirLocation getLocation() { return omEvaluatorValueGetLoc(value); }

  // Get the field location info.
  MlirLocation getFieldLoc(const std::string &name) {
    // Wrap the requested field name in an attribute.
    MlirContext context = mlirTypeGetContext(omEvaluatorObjectGetType(value));
    MlirStringRef cName = mlirStringRefCreateFromCString(name.c_str());
    MlirAttribute nameAttr = mlirStringAttrGet(context, cName);

    // Get the field's ObjectValue via the CAPI.
    OMEvaluatorValue result = omEvaluatorObjectGetField(value, nameAttr);

    return omEvaluatorValueGetLoc(result);
  }

  // Get a field from the Object, using nanobind's support for variant to return
  // a Python object that is either an Object or Attribute.
  PythonValue getField(const std::string &name) {
    // Wrap the requested field name in an attribute.
    MlirContext context = mlirTypeGetContext(omEvaluatorObjectGetType(value));
    MlirStringRef cName = mlirStringRefCreateFromCString(name.c_str());
    MlirAttribute nameAttr = mlirStringAttrGet(context, cName);

    // Get the field's ObjectValue via the CAPI.
    OMEvaluatorValue result = omEvaluatorObjectGetField(value, nameAttr);

    return omEvaluatorValueToPythonValue(result);
  }

  // Get a list with the names of all the fields in the Object.
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

  // Get the hash of the object
  unsigned getHash() { return omEvaluatorObjectGetHash(value); }

  // Check the equality of the underlying values.
  bool eq(Object &other) { return omEvaluatorObjectIsEq(value, other.value); }

  OMEvaluatorValue getValue() const { return value; }

private:
  // The underlying CAPI OMObject.
  OMEvaluatorValue value;
};

/// Provides an Evaluator class by simply wrapping the OMEvaluator CAPI.
struct Evaluator {
  // Instantiate an Evaluator with a reference to the underlying OMEvaluator.
  Evaluator(MlirModule mod) : evaluator(omEvaluatorNew(mod)) {}

  // Instantiate an Object.
  Object instantiate(MlirAttribute className,
                     std::vector<PythonValue> actualParams) {
    std::vector<OMEvaluatorValue> values;
    for (auto &param : actualParams)
      values.push_back(pythonValueToOMEvaluatorValue(
          param, mlirModuleGetContext(getModule())));

    // Instantiate the Object via the CAPI.
    OMEvaluatorValue result = omEvaluatorInstantiate(
        evaluator, className, values.size(), values.data());

    // If the Object is null, something failed. Diagnostic handling is
    // implemented in pure Python, so nothing to do here besides throwing an
    // error to halt execution.
    if (omEvaluatorObjectIsNull(result))
      throw nb::value_error(
          "unable to instantiate object, see previous error(s)");

    // Return a new Object.
    return Object(result);
  }

  // Get the Module the Evaluator is built from.
  MlirModule getModule() { return omEvaluatorGetModule(evaluator); }

private:
  // The underlying CAPI OMEvaluator.
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

class PyMapAttrIterator {
public:
  PyMapAttrIterator(MlirAttribute attr) : attr(std::move(attr)) {}

  PyMapAttrIterator &dunderIter() { return *this; }

  nb::tuple dunderNext() {
    if (nextIndex >= omMapAttrGetNumElements(attr))
      throw nb::stop_iteration();

    MlirIdentifier key = omMapAttrGetElementKey(attr, nextIndex);
    PythonValue value =
        omPrimitiveToPythonValue(omMapAttrGetElementValue(attr, nextIndex));
    nextIndex++;

    auto keyName = mlirIdentifierStr(key);
    std::string keyStr(keyName.data, keyName.length);
    return nb::make_tuple(keyStr, value);
  }

  static void bind(nb::module_ &m) {
    nb::class_<PyMapAttrIterator>(m, "MapAttributeIterator")
        .def("__iter__", &PyMapAttrIterator::dunderIter)
        .def("__next__", &PyMapAttrIterator::dunderNext);
  }

private:
  MlirAttribute attr;
  intptr_t nextIndex = 0;
};

PythonValue Tuple::getElement(intptr_t i) {
  if (i < 0 || i >= omEvaluatorTupleGetNumElements(value))
    throw std::out_of_range("tuple index out of range");

  return omEvaluatorValueToPythonValue(omEvaluatorTupleGetElement(value, i));
}

PythonValue Map::dunderGetItemNamed(const std::string &key) {
  MlirType type = omMapTypeGetKeyType(omEvaluatorMapGetType(value));
  if (!omTypeIsAStringType(type))
    throw nanobind::key_error("key is not string");
  MlirAttribute attr =
      mlirStringAttrTypedGet(type, mlirStringRefCreateFromCString(key.c_str()));
  return dunderGetItemAttr(attr);
}

PythonValue Map::dunderGetItemIndexed(intptr_t i) {
  MlirType type = omMapTypeGetKeyType(omEvaluatorMapGetType(value));
  if (!mlirTypeIsAInteger(type))
    throw nanobind::key_error("key is not integer");
  MlirAttribute attr = mlirIntegerAttrGet(type, i);
  return dunderGetItemAttr(attr);
}

PythonValue Map::dunderGetItemAttr(MlirAttribute key) {
  OMEvaluatorValue result = omEvaluatorMapGetElement(value, key);

  if (omEvaluatorValueIsNull(result))
    throw nanobind::key_error("key not found");

  return omEvaluatorValueToPythonValue(result);
}

PythonValue
Map::dunderGetItem(std::variant<intptr_t, std::string, MlirAttribute> key) {
  if (auto *i = std::get_if<intptr_t>(&key))
    return dunderGetItemIndexed(*i);
  else if (auto *str = std::get_if<std::string>(&key))
    return dunderGetItemNamed(*str);
  return dunderGetItemAttr(std::get<MlirAttribute>(key));
}

// Convert a generic MLIR Attribute to a PythonValue. This is basically a C++
// fast path of the parts of attribute_to_var that we use in the OM dialect.
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

  // BoolAttr's are IntegerAttr's, check this first.
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

  if (omAttrIsAMapAttr(attr)) {
    nb::dict results;
    for (intptr_t i = 0, e = omMapAttrGetNumElements(attr); i < e; ++i) {
      auto keyStrRef = mlirIdentifierStr(omMapAttrGetElementKey(attr, i));
      auto key = nb::str(keyStrRef.data, keyStrRef.length);
      auto value = omPrimitiveToPythonValue(omMapAttrGetElementValue(attr, i));
      results[key] = value;
    }
    return results;
  }

  mlirAttributeDump(attr);
  throw nb::type_error("Unexpected OM primitive attribute");
}

// Convert a primitive PythonValue to a generic MLIR Attribute. This is
// basically a C++ fast path of the parts of var_to_attribute that we use in the
// OM dialect.
static MlirAttribute omPythonValueToPrimitive(PythonPrimitive value,
                                              MlirContext ctx) {
  if (auto *intValue = std::get_if<nb::int_>(&value)) {
    auto intType = mlirIntegerTypeGet(ctx, 64);
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
    return mlirStringAttrGet(ctx, strRef);
  }

  if (auto *attr = std::get_if<nb::bool_>(&value)) {
    return mlirBoolAttrGet(ctx, nb::cast<bool>(*attr));
  }

  throw nb::type_error("Unexpected OM primitive value");
}

PythonValue omEvaluatorValueToPythonValue(OMEvaluatorValue result) {
  // If the result is null, something failed. Diagnostic handling is
  // implemented in pure Python, so nothing to do here besides throwing an
  // error to halt execution.
  if (omEvaluatorValueIsNull(result))
    throw nb::value_error("unable to get field, see previous error(s)");

  // If the field was an Object, return a new Object.
  if (omEvaluatorValueIsAObject(result))
    return Object(result);

  // If the field was a list, return a new List.
  if (omEvaluatorValueIsAList(result))
    return List(result);

  // If the field was a tuple, return a new Tuple.
  if (omEvaluatorValueIsATuple(result))
    return Tuple(result);

  // If the field was a map, return a new Map.
  if (omEvaluatorValueIsAMap(result))
    return Map(result);

  // If the field was a base path, return a new BasePath.
  if (omEvaluatorValueIsABasePath(result))
    return BasePath(result);

  // If the field was a path, return a new Path.
  if (omEvaluatorValueIsAPath(result))
    return Path(result);

  if (omEvaluatorValueIsAReference(result))
    return omEvaluatorValueToPythonValue(
        omEvaluatorValueGetReferenceValue(result));

  // If the field was a primitive, return the Attribute.
  assert(omEvaluatorValueIsAPrimitive(result));
  return omPrimitiveToPythonValue(omEvaluatorValueGetPrimitive(result));
}

OMEvaluatorValue pythonValueToOMEvaluatorValue(PythonValue result,
                                               MlirContext ctx) {
  if (auto *list = std::get_if<List>(&result))
    return list->getValue();

  if (auto *tuple = std::get_if<Tuple>(&result))
    return tuple->getValue();

  if (auto *map = std::get_if<Map>(&result))
    return map->getValue();

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

/// Populate the OM Python module.
void circt::python::populateDialectOMSubmodule(nb::module_ &m) {
  m.doc() = "OM dialect Python native extension";

  // Add the Evaluator class definition.
  nb::class_<Evaluator>(m, "Evaluator")
      .def(nb::init<MlirModule>(), nb::arg("module"))
      .def("instantiate", &Evaluator::instantiate, "Instantiate an Object",
           nb::arg("class_name"), nb::arg("actual_params"))
      .def_prop_ro("module", &Evaluator::getModule,
                   "The Module the Evaluator is built from");

  // Add the List class definition.
  nb::class_<List>(m, "List")
      .def(nb::init<List>(), nb::arg("list"))
      .def("__getitem__", &List::getElement)
      .def("__len__", &List::getNumElements);

  nb::class_<Tuple>(m, "Tuple")
      .def(nb::init<Tuple>(), nb::arg("tuple"))
      .def("__getitem__", &Tuple::getElement)
      .def("__len__", &Tuple::getNumElements);

  // Add the Map class definition.
  nb::class_<Map>(m, "Map")
      .def(nb::init<Map>(), nb::arg("map"))
      .def("__getitem__", &Map::dunderGetItem)
      .def("keys", &Map::getKeys)
      .def_prop_ro("type", &Map::getType, "The Type of the Map");

  // Add the BasePath class definition.
  nb::class_<BasePath>(m, "BasePath")
      .def(nb::init<BasePath>(), nb::arg("basepath"))
      .def_static("get_empty", &BasePath::getEmpty,
                  nb::arg("context") = nb::none());

  // Add the Path class definition.
  nb::class_<Path>(m, "Path")
      .def(nb::init<Path>(), nb::arg("path"))
      .def("__str__", &Path::dunderStr);

  // Add the Object class definition.
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

  // Add the ReferenceAttr definition
  mlir_attribute_subclass(m, "ReferenceAttr", omAttrIsAReferenceAttr)
      .def_property_readonly("inner_ref", [](MlirAttribute self) {
        return omReferenceAttrGetInnerRef(self);
      });

  // Add the IntegerAttr definition
  mlir_attribute_subclass(m, "OMIntegerAttr", omAttrIsAIntegerAttr)
      .def_classmethod("get",
                       [](nb::object cls, MlirAttribute intVal) {
                         return cls(omIntegerAttrGet(intVal));
                       })
      .def_property_readonly(
          "integer",
          [](MlirAttribute self) { return omIntegerAttrGetInt(self); })
      .def("__str__", [](MlirAttribute self) {
        MlirStringRef str = omIntegerAttrToString(self);
        return std::string(str.data, str.length);
      });

  // Add the OMListAttr definition
  mlir_attribute_subclass(m, "ListAttr", omAttrIsAListAttr)
      .def("__getitem__", &omListAttrGetElement)
      .def("__len__", &omListAttrGetNumElements)
      .def("__iter__",
           [](MlirAttribute arr) { return PyListAttrIterator(arr); });
  PyListAttrIterator::bind(m);

  // Add the MapAttr definition
  mlir_attribute_subclass(m, "MapAttr", omAttrIsAMapAttr)
      .def("__iter__", [](MlirAttribute arr) { return PyMapAttrIterator(arr); })
      .def("__len__", &omMapAttrGetNumElements);
  PyMapAttrIterator::bind(m);

  // Add the AnyType class definition.
  mlir_type_subclass(m, "AnyType", omTypeIsAAnyType, omAnyTypeGetTypeID);

  // Add the ClassType class definition.
  mlir_type_subclass(m, "ClassType", omTypeIsAClassType, omClassTypeGetTypeID)
      .def_property_readonly("name", [](MlirType type) {
        MlirStringRef name = mlirIdentifierStr(omClassTypeGetName(type));
        return std::string(name.data, name.length);
      });

  // Add the BasePathType class definition.
  mlir_type_subclass(m, "BasePathType", omTypeIsAFrozenBasePathType,
                     omFrozenBasePathTypeGetTypeID);

  // Add the ListType class definition.
  mlir_type_subclass(m, "ListType", omTypeIsAListType, omListTypeGetTypeID)
      .def_property_readonly("element_type", omListTypeGetElementType);

  // Add the PathType class definition.
  mlir_type_subclass(m, "PathType", omTypeIsAFrozenPathType,
                     omFrozenPathTypeGetTypeID);
}
