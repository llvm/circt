//===- MLIRPybindAdaptors.h - Adaptors for interop with MLIR APIs ---------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains adaptors for out of tree dialects to use and extend the
// MLIR Python API via CAPI types.
//
// It is being developed out of tree but should be considered a candidate to
// move upstream once stable, so that everyone can use it.
//===----------------------------------------------------------------------===//

#ifndef CIRCT_BINDINGS_PYTHON_MLIRCAPIADAPTORS_H
#define CIRCT_BINDINGS_PYTHON_MLIRCAPIADAPTORS_H

#include <pybind11/pybind11.h>
#include <pybind11/pytypes.h>
#include <pybind11/stl.h>

#include "mlir-c/Bindings/Python/Interop.h"
#include "mlir-c/IR.h"

namespace py = pybind11;

// TODO: Move this to Interop.h and make it externally configurable/use it
// consistently to locate the "import mlir" top-level.
#define MLIR_PYTHON_PACKAGE_PREFIX "mlir."

// Raw CAPI type casters need to be declared before use, so always include them
// first.
namespace pybind11 {
namespace detail {

template <typename T>
struct type_caster<llvm::Optional<T>> : optional_caster<llvm::Optional<T>> {};

/// Helper to convert a presumed MLIR API object to a capsule, accepting either
/// an explicit Capsule (which can happen when two C APIs are communicating
/// directly via Python) or indirectly by querying the MLIR_PYTHON_CAPI_PTR_ATTR
/// attribute (through which supported MLIR Python API objects export their
/// contained API pointer as a capsule). This is intended to be used from
/// type casters, which are invoked with a raw handle (unowned). The returned
/// object's lifetime may not extend beyond the apiObject handle without
/// explicitly having its refcount increased (i.e. on return).
static py::object mlirApiObjectToCapsule(py::handle apiObject) {
  if (PyCapsule_CheckExact(apiObject.ptr()))
    return py::reinterpret_borrow<py::object>(apiObject);
  return apiObject.attr(MLIR_PYTHON_CAPI_PTR_ATTR);
}

// Note: Currently all of the following support cast from py::object to the
// Mlir* C-API type, but only a few light-weight, context-bound ones
// implicitly cast the other way because the use case has not yet emerged and
// ownership is unclear.

/// Casts object -> MlirAttribute.
template <>
struct type_caster<MlirAttribute> {
  PYBIND11_TYPE_CASTER(MlirAttribute, _("MlirAttribute"));
  bool load(handle src, bool) {
    auto capsule = mlirApiObjectToCapsule(src);
    value = mlirPythonCapsuleToAttribute(capsule.ptr());
    if (mlirAttributeIsNull(value)) {
      return false;
    }
    return true;
  }
  static handle cast(MlirAttribute v, return_value_policy, handle) {
    auto capsule =
        py::reinterpret_steal<py::object>(mlirPythonAttributeToCapsule(v));
    return py::module::import("mlir.ir")
        .attr("Attribute")
        .attr(MLIR_PYTHON_CAPI_FACTORY_ATTR)(capsule)
        .release();
  }
};

/// Casts object -> MlirContext.
template <>
struct type_caster<MlirContext> {
  PYBIND11_TYPE_CASTER(MlirContext, _("MlirContext"));
  bool load(handle src, bool) {
    auto capsule = mlirApiObjectToCapsule(src);
    value = mlirPythonCapsuleToContext(capsule.ptr());
    if (mlirContextIsNull(value)) {
      return false;
    }
    return true;
  }
};

/// Casts object -> MlirLocation.
template <>
struct type_caster<MlirLocation> {
  PYBIND11_TYPE_CASTER(MlirLocation, _("MlirLocation"));
  bool load(handle src, bool) {
    auto capsule = mlirApiObjectToCapsule(src);
    value = mlirPythonCapsuleToLocation(capsule.ptr());
    if (mlirLocationIsNull(value)) {
      return false;
    }
    return true;
  }
  static handle cast(MlirLocation v, return_value_policy, handle) {
    auto capsule =
        py::reinterpret_steal<py::object>(mlirPythonLocationToCapsule(v));
    return py::module::import("mlir.ir")
        .attr("Location")
        .attr(MLIR_PYTHON_CAPI_FACTORY_ATTR)(capsule)
        .release();
  }
};

/// Casts object -> MlirModule.
template <>
struct type_caster<MlirModule> {
  PYBIND11_TYPE_CASTER(MlirModule, _("MlirModule"));
  bool load(handle src, bool) {
    auto capsule = mlirApiObjectToCapsule(src);
    value = mlirPythonCapsuleToModule(capsule.ptr());
    if (mlirModuleIsNull(value)) {
      return false;
    }
    return true;
  }
  static handle cast(MlirModule v, return_value_policy, handle) {
    auto capsule =
        py::reinterpret_steal<py::object>(mlirPythonModuleToCapsule(v));
    return py::module::import("mlir.ir")
        .attr("Module")
        .attr(MLIR_PYTHON_CAPI_FACTORY_ATTR)(capsule)
        .release();
  };
};

/// Casts object <-> MlirOperation.
template <>
struct type_caster<MlirOperation> {
  PYBIND11_TYPE_CASTER(MlirOperation, _("MlirOperation"));
  bool load(handle src, bool) {
    auto capsule = mlirApiObjectToCapsule(src);
    value = mlirPythonCapsuleToOperation(capsule.ptr());
    if (mlirOperationIsNull(value)) {
      return false;
    }
    return true;
  }
  static handle cast(MlirOperation v, return_value_policy, handle) {
    if (v.ptr == nullptr)
      return py::none();
    auto capsule =
        py::reinterpret_steal<py::object>(mlirPythonOperationToCapsule(v));
    return py::module::import("mlir.ir")
        .attr("Operation")
        .attr(MLIR_PYTHON_CAPI_FACTORY_ATTR)(capsule)
        .release();
  };
};

/// Casts object -> MlirPassManager.
template <>
struct type_caster<MlirPassManager> {
  PYBIND11_TYPE_CASTER(MlirPassManager, _("MlirPassManager"));
  bool load(handle src, bool) {
    auto capsule = mlirApiObjectToCapsule(src);
    value = mlirPythonCapsuleToPassManager(capsule.ptr());
    if (mlirPassManagerIsNull(value)) {
      return false;
    }
    return true;
  }
};

/// Casts object -> MlirType.
template <>
struct type_caster<MlirType> {
  PYBIND11_TYPE_CASTER(MlirType, _("MlirType"));
  bool load(handle src, bool) {
    auto capsule = mlirApiObjectToCapsule(src);
    value = mlirPythonCapsuleToType(capsule.ptr());
    if (mlirTypeIsNull(value)) {
      return false;
    }
    return true;
  }
  static handle cast(MlirType t, return_value_policy, handle) {
    auto capsule =
        py::reinterpret_steal<py::object>(mlirPythonTypeToCapsule(t));
    return py::module::import("mlir.ir")
        .attr("Type")
        .attr(MLIR_PYTHON_CAPI_FACTORY_ATTR)(capsule)
        .release();
  }
};

} // namespace detail
} // namespace pybind11

namespace mlir {
namespace python {
namespace adaptors {

/// Provides a facility like py::class_ for defining a new class in a scope,
/// but this allows extension of an arbitrary Python class, defining methods
/// on it is a similar way. Classes defined in this way are very similar to
/// if defined in Python in the usual way but use Pybind11 machinery to do
/// it. These are not "real" Pybind11 classes but pure Python classes with no
/// relation to a concrete C++ class.
///
/// Derived from a discussion upstream:
///   https://github.com/pybind/pybind11/issues/1193
///   (plus a fair amount of extra curricular poking)
///   TODO: If this proves useful, see about including it in pybind11.
class pure_subclass {
public:
  struct super_class_info {
    const char *className;
    const char *moduleName = MLIR_PYTHON_PACKAGE_PREFIX "ir";
  };
  pure_subclass(py::handle scope, const char *derivedClassName,
                super_class_info superClassInfo) {
    superClass = py::module::import(superClassInfo.moduleName)
                     .attr(superClassInfo.className);
    py::object pyType =
        py::reinterpret_borrow<py::object>((PyObject *)&PyType_Type);
    py::object metaclass = pyType(superClass);
    py::dict attributes;

    thisClass =
        metaclass(derivedClassName, py::make_tuple(superClass), attributes);
    scope.attr(derivedClassName) = thisClass;
  }

  template <typename Func, typename... Extra>
  pure_subclass &def(const char *name, Func &&f, const Extra &... extra) {
    py::cpp_function cf(
        std::forward<Func>(f), py::name(name), py::is_method(py::none()),
        py::sibling(py::getattr(thisClass, name, py::none())), extra...);
    thisClass.attr(cf.name()) = cf;
    return *this;
  }

  template <typename Func, typename... Extra>
  pure_subclass &def_property_readonly(const char *name, Func &&f,
                                       const Extra &... extra) {
    py::cpp_function cf(
        std::forward<Func>(f), py::name(name), py::is_method(py::none()),
        py::sibling(py::getattr(thisClass, name, py::none())), extra...);
    auto builtinProperty =
        py::reinterpret_borrow<py::object>((PyObject *)&PyProperty_Type);
    thisClass.attr(name) = builtinProperty(cf);
    return *this;
  }

  template <typename Func, typename... Extra>
  pure_subclass &def_static(const char *name, Func &&f,
                            const Extra &... extra) {
    static_assert(
        !std::is_member_function_pointer<Func>::value,
        "def_static(...) called with a non-static member function pointer");
    py::cpp_function cf(
        std::forward<Func>(f), py::name(name), py::scope(thisClass),
        py::sibling(py::getattr(thisClass, name, py::none())), extra...);
    thisClass.attr(cf.name()) = py::staticmethod(cf);
    return *this;
  }

protected:
  py::object superClass;
  py::object thisClass;
};

class mlir_type_subclass : public pure_subclass {
public:
  using IsAFunctionTy = bool (*)(MlirType);

  mlir_type_subclass(py::handle scope, const char *typeClassName,
                     IsAFunctionTy isaFunction)
      : mlir_type_subclass(scope, typeClassName, isaFunction, {"Type"}) {}

  mlir_type_subclass(py::handle scope, const char *typeClassName,
                     IsAFunctionTy isaFunction, super_class_info superClassInfo)
      : pure_subclass(scope, typeClassName, superClassInfo) {
    // Casting constructor. Note that defining an __init__ method is special
    // and not yet generalized on pure_subclass (it requires a somewhat
    // different cpp_function and other requirements on chaining to super
    // __init__ make it more awkward to do generally).
    std::string captureTypeName(
        typeClassName); // As string in case if typeClassName is not static.
    py::object captureSuperClass = superClass;
    py::cpp_function initCf(
        [captureSuperClass, isaFunction,
         captureTypeName](py::object self, py::object otherType) {
          MlirType rawType = py::cast<MlirType>(otherType);
          if (!isaFunction(rawType)) {
            auto origRepr = py::repr(otherType).cast<std::string>();
            throw std::invalid_argument((llvm::Twine("Cannot cast type to ") +
                                         captureTypeName + " (from " +
                                         origRepr + ")")
                                            .str());
          }
          captureSuperClass.attr("__init__")(self, otherType);
        },
        py::arg("cast_from_type"), py::is_method(py::none()),
        "Casts the passed type to this specific sub-type.");
    thisClass.attr("__init__") = initCf;

    // 'isinstance' method.
    def_static(
        "isinstance",
        [isaFunction](MlirType other) { return isaFunction(other); },
        py::arg("other_type"));
  }
};

} // namespace adaptors
} // namespace python
} // namespace mlir

#endif // CIRCT_BINDINGS_PYTHON_PYBINDUTILS_H
