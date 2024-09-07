//===- SupportModule.cpp - Support API pybind module ----------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "CIRCTModules.h"

#include "mlir/Bindings/Python/PybindAdaptors.h"

#include "PybindUtils.h"
#include "mlir-c/Support.h"
#include <pybind11/pybind11.h>
#include <pybind11/pytypes.h>
#include <pybind11/stl.h>

namespace py = pybind11;

using namespace circt;
using namespace mlir::python::adaptors;

/// Populate the support python module.
void circt::python::populateSupportSubmodule(py::module &m) {
  m.doc() = "CIRCT Python utils";
  // Walk with filter.
  m.def(
      "_walk_with_filter",
      [](MlirOperation operation, std::vector<std::string> op_names,
         std::function<MlirWalkResult(MlirOperation)> callback,
         MlirWalkOrder walkOrder) {
        struct UserData {
          std::function<MlirWalkResult(MlirOperation)> callback;
          bool gotException;
          std::string exceptionWhat;
          py::object exceptionType;
          std::vector<MlirIdentifier> op_names;
        };

        std::vector<MlirIdentifier> op_names_identifiers;

        // Construct MlirIdentifier from string to perform pointer comparison.
        for (auto &op_name : op_names)
          op_names_identifiers.push_back(mlirIdentifierGet(
              mlirOperationGetContext(operation),
              mlirStringRefCreateFromCString(op_name.c_str())));

        UserData userData{callback, false, {}, {}, op_names_identifiers};
        MlirOperationWalkCallback walkCallback = [](MlirOperation op,
                                                    void *userData) {
          UserData *calleeUserData = static_cast<UserData *>(userData);
          auto op_name = mlirOperationGetName(op);

          // Check if the operation name is in the filter.
          bool inFilter = false;
          for (auto &op_name_identifier : calleeUserData->op_names) {
            if (mlirIdentifierEqual(op_name, op_name_identifier)) {
              inFilter = true;
              break;
            }
          }

          // If the operation name is not in the filter, skip it.
          if (!inFilter)
            return MlirWalkResult::MlirWalkResultAdvance;

          try {
            return (calleeUserData->callback)(op);
          } catch (py::error_already_set &e) {
            calleeUserData->gotException = true;
            calleeUserData->exceptionWhat = e.what();
            calleeUserData->exceptionType = e.type();
            return MlirWalkResult::MlirWalkResultInterrupt;
          }
        };
        mlirOperationWalk(operation, walkCallback, &userData, walkOrder);
        if (userData.gotException) {
          std::string message("Exception raised in callback: ");
          message.append(userData.exceptionWhat);
          throw std::runtime_error(message);
        }
      },
      py::arg("op"), py::arg("op_names"), py::arg("callback"),
      py::arg("walk_order"));
}
