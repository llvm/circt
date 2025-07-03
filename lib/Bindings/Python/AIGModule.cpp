//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "CIRCTModules.h"

#include "circt-c/Dialect/AIG.h"

#include "mlir-c/Support.h"
#include "mlir/Bindings/Python/NanobindAdaptors.h"
#include "mlir/CAPI/Wrap.h"

#include "NanobindUtils.h"
#include <nanobind/nanobind.h>
#include <string_view>

namespace nb = nanobind;

using namespace circt;
using namespace mlir::python::nanobind_adaptors;

/// Populate the aig python module.
void circt::python::populateDialectAIGSubmodule(nb::module_ &m) {
  m.doc() = "AIG dialect Python native extension";

  // LongestPathAnalysis class
  nb::class_<AIGLongestPathAnalysis>(m, "_LongestPathAnalysis")
      .def(
          "__init__",
          [](AIGLongestPathAnalysis *self, MlirOperation module,
             bool traceDebugPoints) {
            new (self) AIGLongestPathAnalysis(
                aigLongestPathAnalysisCreate(module, traceDebugPoints));
          },
          nb::arg("module"), nb::arg("trace_debug_points") = true)
      .def("__del__",
           [](AIGLongestPathAnalysis &self) {
             aigLongestPathAnalysisDestroy(self);
           })
      .def("get_all_paths",
           [](AIGLongestPathAnalysis *self, const std::string &moduleName,
              bool elaboratePaths) -> AIGLongestPathCollection {
             MlirStringRef moduleNameRef =
                 mlirStringRefCreateFromCString(moduleName.c_str());

             if (aigLongestPathCollectionIsNull(
                     aigLongestPathAnalysisGetAllPaths(*self, moduleNameRef,
                                                       elaboratePaths)))
               throw nb::value_error(
                   "Failed to get all paths, see previous error(s).");

             auto collection =
                 AIGLongestPathCollection(aigLongestPathAnalysisGetAllPaths(
                     *self, moduleNameRef, elaboratePaths));
             return collection;
           });

  nb::class_<AIGLongestPathCollection>(m, "_LongestPathCollection")
      .def("__del__",
           [](AIGLongestPathCollection &self) {
             aigLongestPathCollectionDestroy(self);
           })
      .def("get_size",
           [](AIGLongestPathCollection &self) {
             return aigLongestPathCollectionGetSize(self);
           })
      .def("get_path",
           [](AIGLongestPathCollection &self,
              int pathIndex) -> std::string_view {
             MlirStringRef pathRef =
                 aigLongestPathCollectionGetPath(self, pathIndex);
             return std::string_view(pathRef.data, pathRef.length);
           });
}
