//===- RTLModule.cpp - RTL dialect pybind module --------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "RTLModule.h"
#include "IRModule.h"
#include "circt-c/Dialect/RTL.h"
#include "mlir-c/Bindings/Python/Interop.h"
#include "mlir/Support/LLVM.h"

using namespace mlir;
using namespace mlir::python;

namespace py = pybind11;

/// TODO(mikeurbach): expose this upstream.
/// CRTP base classes for Python types that subclass Type and should be
/// castable from it (i.e. via something like IntegerType(t)).
/// By default, type class hierarchies are one level deep (i.e. a
/// concrete type class extends PyType); however, intermediate python-visible
/// base classes can be modeled by specifying a BaseTy.
template <typename DerivedTy, typename BaseTy = PyType>
class PyConcreteType : public BaseTy {
public:
  // Derived classes must define statics for:
  //   IsAFunctionTy isaFunction
  //   const char *pyClassName
  using ClassTy = py::class_<DerivedTy, BaseTy>;
  using IsAFunctionTy = bool (*)(MlirType);

  PyConcreteType() = default;
  PyConcreteType(PyMlirContextRef contextRef, MlirType t)
      : BaseTy(std::move(contextRef), t) {}
  PyConcreteType(BaseTy &orig)
      : PyConcreteType(orig.getContext(), castFrom(orig)) {}

  static MlirType castFrom(BaseTy &orig) {
    if (!DerivedTy::isaFunction(orig)) {
      auto origRepr = py::repr(py::cast(orig)).template cast<std::string>();
      // TODO(mikeurbach): this needs to be marked MLIR_CAPI_EXPORTED upstream
      // for us to be able to depend on _mlir directly.
      // throw SetPyError(PyExc_ValueError, Twine("Cannot cast type to ") +
      //                                        DerivedTy::pyClassName +
      //                                        " (from " + origRepr + ")");
    }
    return orig;
  }

  static void bind(py::module &m) {
    auto cls = ClassTy(m, DerivedTy::pyClassName);
    cls.def(py::init<BaseTy &>(), py::keep_alive<0, 1>());
    cls.def_static("isinstance", [](BaseTy &otherType) -> bool {
      return DerivedTy::isaFunction(otherType);
    });
    DerivedTy::bindDerived(cls);
  }

  /// Implemented by derived classes to add methods to the Python subclass.
  static void bindDerived(ClassTy &m) {}
};

class PyInOutType : public PyConcreteType<PyInOutType> {
public:
  static constexpr const char *pyClassName = "InOutType";
  static constexpr IsAFunctionTy isaFunction = rtlTypeIsAInOut;
  using PyConcreteType::PyConcreteType;

  static void bindDerived(ClassTy &c) {
    c.def_static(
        "get",
        [](PyType elementType) {
          MlirType mlirElementType = elementType.get();
          MlirType mlirInOutType = rtlInOutTypeGet(mlirElementType);
          return PyInOutType(elementType.getContext(), mlirInOutType);
        },
        py::arg("element_type"), "Create an InOutType from an element Type.");
  }
};

void populateRTLModule(pybind11::module &m) { PyInOutType::bind(m); }
