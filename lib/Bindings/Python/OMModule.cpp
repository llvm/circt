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
#include "mlir-c/IR.h"
#include "mlir/Bindings/Python/PybindAdaptors.h"
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
namespace py = pybind11;

using namespace mlir;
using namespace mlir::python;
using namespace mlir::python::adaptors;

namespace {

struct Object;
struct List;

using PythonValue = std::variant<MlirAttribute, Object, List>;

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

/// Provides an Object class by simply wrapping the OMObject CAPI.
struct Object {
  // Instantiate an Object with a reference to the underlying OMObject.
  Object(OMEvaluatorValue value) : value(value) {}

  /// Get the Type from an Object, which will be a ClassType.
  MlirType getType() { return omEvaluatorObjectGetType(value); }

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

  // If the field was a primitive, return the Attribute.
  assert(omEvaluatorValueIsAPrimitive(result));
  return omEvaluatorValueGetPrimitive(result);
}

OMEvaluatorValue pythonValueToOMEvaluatorValue(PythonValue result) {
  if (auto *attr = std::get_if<MlirAttribute>(&result))
    return omEvaluatorValueFromPrimitive(*attr);

  if (auto *list = std::get_if<List>(&result))
    return list->getValue();

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

  // Add the Object class definition.
  py::class_<Object>(m, "Object")
      .def(py::init<Object>(), py::arg("object"))
      .def("__getattr__", &Object::getField, "Get a field from an Object",
           py::arg("name"))
      .def_property_readonly("field_names", &Object::getFieldNames,
                             "Get field names from an Object")
      .def_property_readonly("type", &Object::getType,
                             "The Type of the Object");

  // Add the ReferenceAttr definition
  mlir_attribute_subclass(m, "ReferenceAttr", omAttrIsAReferenceAttr)
      .def_property_readonly("inner_ref", [](MlirAttribute self) {
        return omReferenceAttrGetInnerRef(self);
      });

  // Add the OMListAttr definition
  mlir_attribute_subclass(m, "ListAttr", omAttrIsAListAttr)
      .def("__getitem__", &omListAttrGetElement)
      .def("__len__", &omListAttrGetNumElements)
      .def("__iter__",
           [](MlirAttribute arr) { return PyListAttrIterator(arr); });
  PyListAttrIterator::bind(m);

  // Add the ClassType class definition.
  mlir_type_subclass(m, "ClassType", omTypeIsAClassType);
}
