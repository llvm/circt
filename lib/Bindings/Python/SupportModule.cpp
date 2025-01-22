//===- SupportModule.cpp - Support API nanobind module --------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "CIRCTModules.h"

#include "mlir/Bindings/Python/NanobindAdaptors.h"

#include "NanobindUtils.h"
#include "mlir-c/Support.h"
#include <nanobind/nanobind.h>

namespace nb = nanobind;

using namespace circt;
using namespace mlir::python::nanobind_adaptors;

/// Populate the support python module.
void circt::python::populateSupportSubmodule(nb::module_ &m) {
  m.doc() = "CIRCT Python utils";
  // Walk with filter.
  m.def(
      "_walk_with_filter",
      [](MlirOperation operation, const std::vector<std::string> &opNames,
         std::function<nb::object(MlirOperation)> callback,
         nb::object walkOrderRaw) {
        struct UserData {
          std::function<nb::object(MlirOperation)> callback;
          bool gotException;
          std::string exceptionWhat;
          nb::handle exceptionType;
          std::vector<MlirIdentifier> opNames;
        };

        // As we transition from nanobind to nanobind, the WalkOrder enum and
        // automatic casting will be defined as a nanobind enum upstream. Do a
        // manual conversion that works with either nanobind or nanobind for
        // now. When we're on nanobind in CIRCT, we can go back to automatic
        // casting.
        MlirWalkOrder walkOrder;
        auto walkOrderRawValue = nb::cast<int>(walkOrderRaw.attr("value"));
        switch (walkOrderRawValue) {
        case 0:
          walkOrder = MlirWalkOrder::MlirWalkPreOrder;
          break;
        case 1:
          walkOrder = MlirWalkOrder::MlirWalkPostOrder;
          break;
        }

        std::vector<MlirIdentifier> opNamesIdentifiers;
        opNamesIdentifiers.reserve(opNames.size());

        // Construct MlirIdentifier from string to perform pointer comparison.
        for (auto &opName : opNames)
          opNamesIdentifiers.push_back(mlirIdentifierGet(
              mlirOperationGetContext(operation),
              mlirStringRefCreateFromCString(opName.c_str())));

        UserData userData{
            std::move(callback), false, {}, {}, opNamesIdentifiers};
        MlirOperationWalkCallback walkCallback = [](MlirOperation op,
                                                    void *userData) {
          UserData *calleeUserData = static_cast<UserData *>(userData);
          auto opName = mlirOperationGetName(op);

          // Check if the operation name is in the filter.
          bool inFilter = false;
          for (auto &opNamesIdentifier : calleeUserData->opNames) {
            if (mlirIdentifierEqual(opName, opNamesIdentifier)) {
              inFilter = true;
              break;
            }
          }

          // If the operation name is not in the filter, skip it.
          if (!inFilter)
            return MlirWalkResult::MlirWalkResultAdvance;

          try {
            // As we transition from nanobind to nanobind, the WalkResult enum
            // and automatic casting will be defined as a nanobind enum
            // upstream. Do a manual conversion that works with either nanobind
            // or nanobind for now. When we're on nanobind in CIRCT, we can go
            // back to automatic casting.
            MlirWalkResult walkResult;
            auto walkResultRaw = (calleeUserData->callback)(op);
            auto walkResultRawValue =
                nb::cast<int>(walkResultRaw.attr("value"));
            switch (walkResultRawValue) {
            case 0:
              walkResult = MlirWalkResult::MlirWalkResultAdvance;
              break;
            case 1:
              walkResult = MlirWalkResult::MlirWalkResultInterrupt;
              break;
            case 2:
              walkResult = MlirWalkResult::MlirWalkResultSkip;
              break;
            }
            return walkResult;
          } catch (nb::python_error &e) {
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
      nb::arg("op"), nb::arg("op_names"), nb::arg("callback"),
      nb::arg("walk_order"));
}
