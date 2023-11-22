//===- OMModule.cpp - OM API pybind module --------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "DialectModules.h"
#include "circt-c/Dialect/OM.h"
#include "mlir-c/BuiltinAttributes.h"
#include "mlir-c/BuiltinTypes.h"
#include "mlir-c/IR.h"
#include "mlir/Bindings/Python/PybindAdaptors.h"
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
namespace py = pybind11;

using namespace mlir;
using namespace mlir::python;
using namespace mlir::python::adaptors;

namespace {

struct List;
struct Object;
struct Tuple;
struct Map;
struct BasePath;
struct Path;

/// None is used to by pybind when default initializing a PythonValue. The order
/// of types in the variant matters here, and we want pybind to try casting to
/// the Python classes defined in this file first, before MlirAttribute and the
/// upstream MLIR type casters.  If the MlirAttribute is tried first, then we
/// can hit an assert inside the MLIR codebase.
struct None {};
using PythonValue =
    std::variant<None, Object, List, Tuple, Map, BasePath, Path, MlirAttribute>;

/// Map an opaque OMEvaluatorValue into a python value.
PythonValue omEvaluatorValueToPythonValue(OMEvaluatorValue result);
OMEvaluatorValue pythonValueToOMEvaluatorValue(PythonValue result);

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
  std::vector<MlirAttribute> getKeys() {
    auto attr = omEvaluatorMapGetKeys(value);
    intptr_t numFieldNames = mlirArrayAttrGetNumElements(attr);

    std::vector<MlirAttribute> pyFieldNames;
    for (intptr_t i = 0; i < numFieldNames; ++i)
      pyFieldNames.emplace_back(mlirArrayAttrGetElement(attr, i));

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

  // Get a field from the Object, using pybind's support for variant to return a
  // Python object that is either an Object or Attribute.
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
      values.push_back(pythonValueToOMEvaluatorValue(param));

    // Instantiate the Object via the CAPI.
    OMEvaluatorValue result = omEvaluatorInstantiate(
        evaluator, className, values.size(), values.data());

    // If the Object is null, something failed. Diagnostic handling is
    // implemented in pure Python, so nothing to do here besides throwing an
    // error to halt execution.
    if (omEvaluatorObjectIsNull(result))
      throw py::value_error(
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
      throw py::stop_iteration();
    return omListAttrGetElement(attr, nextIndex++);
  }

  static void bind(py::module &m) {
    py::class_<PyListAttrIterator>(m, "ListAttributeIterator",
                                   py::module_local())
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

  py::tuple dunderNext() {
    if (nextIndex >= omMapAttrGetNumElements(attr))
      throw py::stop_iteration();

    MlirIdentifier key = omMapAttrGetElementKey(attr, nextIndex);
    MlirAttribute value = omMapAttrGetElementValue(attr, nextIndex);
    nextIndex++;

    auto keyName = mlirIdentifierStr(key);
    std::string keyStr(keyName.data, keyName.length);
    return py::make_tuple(keyStr, value);
  }

  static void bind(py::module &m) {
    py::class_<PyMapAttrIterator>(m, "MapAttributeIterator", py::module_local())
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
    throw pybind11::key_error("key is not string");
  MlirAttribute attr =
      mlirStringAttrTypedGet(type, mlirStringRefCreateFromCString(key.c_str()));
  return dunderGetItemAttr(attr);
}

PythonValue Map::dunderGetItemIndexed(intptr_t i) {
  MlirType type = omMapTypeGetKeyType(omEvaluatorMapGetType(value));
  if (!mlirTypeIsAInteger(type))
    throw pybind11::key_error("key is not integer");
  MlirAttribute attr = mlirIntegerAttrGet(type, i);
  return dunderGetItemAttr(attr);
}

PythonValue Map::dunderGetItemAttr(MlirAttribute key) {
  OMEvaluatorValue result = omEvaluatorMapGetElement(value, key);

  if (omEvaluatorValueIsNull(result))
    throw pybind11::key_error("key not found");

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

PythonValue omEvaluatorValueToPythonValue(OMEvaluatorValue result) {
  // If the result is null, something failed. Diagnostic handling is
  // implemented in pure Python, so nothing to do here besides throwing an
  // error to halt execution.
  if (omEvaluatorValueIsNull(result))
    throw py::value_error("unable to get field, see previous error(s)");

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

  // If the field was a primitive, return the Attribute.
  assert(omEvaluatorValueIsAPrimitive(result));
  return omEvaluatorValueGetPrimitive(result);
}

OMEvaluatorValue pythonValueToOMEvaluatorValue(PythonValue result) {
  if (auto *attr = std::get_if<MlirAttribute>(&result))
    return omEvaluatorValueFromPrimitive(*attr);

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

  return std::get<Object>(result).getValue();
}

} // namespace

/// Populate the OM Python module.
void circt::python::populateDialectOMSubmodule(py::module &m) {
  m.doc() = "OM dialect Python native extension";

  // Add the Evaluator class definition.
  py::class_<Evaluator>(m, "Evaluator")
      .def(py::init<MlirModule>(), py::arg("module"))
      .def("instantiate", &Evaluator::instantiate, "Instantiate an Object",
           py::arg("class_name"), py::arg("actual_params"))
      .def_property_readonly("module", &Evaluator::getModule,
                             "The Module the Evaluator is built from");

  // Add the List class definition.
  py::class_<List>(m, "List")
      .def(py::init<List>(), py::arg("list"))
      .def("__getitem__", &List::getElement)
      .def("__len__", &List::getNumElements);

  py::class_<Tuple>(m, "Tuple")
      .def(py::init<Tuple>(), py::arg("tuple"))
      .def("__getitem__", &Tuple::getElement)
      .def("__len__", &Tuple::getNumElements);

  // Add the Map class definition.
  py::class_<Map>(m, "Map")
      .def(py::init<Map>(), py::arg("map"))
      .def("__getitem__", &Map::dunderGetItem)
      .def("keys", &Map::getKeys)
      .def_property_readonly("type", &Map::getType, "The Type of the Map");

  // Add the BasePath class definition.
  py::class_<BasePath>(m, "BasePath")
      .def(py::init<BasePath>(), py::arg("basepath"))
      .def_static("get_empty", &BasePath::getEmpty,
                  py::arg("context") = py::none());

  // Add the Path class definition.
  py::class_<Path>(m, "Path")
      .def(py::init<Path>(), py::arg("path"))
      .def("__str__", &Path::dunderStr);

  // Add the Object class definition.
  py::class_<Object>(m, "Object")
      .def(py::init<Object>(), py::arg("object"))
      .def("__getattr__", &Object::getField, "Get a field from an Object",
           py::arg("name"))
      .def("get_field_loc", &Object::getFieldLoc,
           "Get the location of a field from an Object", py::arg("name"))
      .def_property_readonly("field_names", &Object::getFieldNames,
                             "Get field names from an Object")
      .def_property_readonly("type", &Object::getType, "The Type of the Object")
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
                       [](py::object cls, MlirAttribute intVal) {
                         return cls(omIntegerAttrGet(intVal));
                       })
      .def_property_readonly("integer", [](MlirAttribute self) {
        return omIntegerAttrGetInt(self);
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

  // Add the ClassType class definition.
  mlir_type_subclass(m, "ClassType", omTypeIsAClassType, omClassTypeGetTypeID)
      .def_property_readonly("name", [](MlirType type) {
        MlirStringRef name = mlirIdentifierStr(omClassTypeGetName(type));
        return std::string(name.data, name.length);
      });

  // Add the BasePathType class definition.
  mlir_type_subclass(m, "BasePathType", omTypeIsAFrozenBasePathType,
                     omFrozenBasePathTypeGetTypeID);

  // Add the PathType class definition.
  mlir_type_subclass(m, "PathType", omTypeIsAFrozenPathType,
                     omFrozenPathTypeGetTypeID);
}
