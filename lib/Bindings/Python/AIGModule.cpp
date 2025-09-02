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
             bool collectDebugInfo, bool keepOnlyMaxDelayPaths,
             bool lazyComputation) {
            new (self) AIGLongestPathAnalysis(aigLongestPathAnalysisCreate(
                module, collectDebugInfo, keepOnlyMaxDelayPaths,
                lazyComputation));
          },
          nb::arg("module"), nb::arg("collect_debug_info") = false,
          nb::arg("keep_only_max_delay_paths") = false,
          nb::arg("lazy_computation") = false)
      .def("__del__",
           [](AIGLongestPathAnalysis &self) {
             aigLongestPathAnalysisDestroy(self);
           })
      .def("get_paths",
           [](AIGLongestPathAnalysis *self, MlirValue value, int64_t bitPos,
              bool elaboratePaths) -> AIGLongestPathCollection {
             auto collection =
                 AIGLongestPathCollection(aigLongestPathAnalysisGetPaths(
                     *self, value, bitPos, elaboratePaths));

             if (aigLongestPathCollectionIsNull(collection))
               throw nb::value_error(
                   "Failed to get all paths, see previous error(s).");

             return collection;
           })
      .def("get_all_paths",
           [](AIGLongestPathAnalysis *self, const std::string &moduleName,
              bool elaboratePaths) -> AIGLongestPathCollection {
             MlirStringRef moduleNameRef =
                 mlirStringRefCreateFromCString(moduleName.c_str());

             auto collection =
                 AIGLongestPathCollection(aigLongestPathAnalysisGetAllPaths(
                     *self, moduleNameRef, elaboratePaths));

             if (aigLongestPathCollectionIsNull(collection))
               throw nb::value_error(
                   "Failed to get all paths, see previous error(s).");

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
              int pathIndex) -> AIGLongestPathDataflowPath {
             return aigLongestPathCollectionGetDataflowPath(self, pathIndex);
           })
      .def("merge",
           [](AIGLongestPathCollection &self, AIGLongestPathCollection &src) {
             aigLongestPathCollectionMerge(self, src);
           });

  nb::class_<AIGLongestPathDataflowPath>(m, "_LongestPathDataflowPath")
      .def_prop_ro("delay",
                   [](AIGLongestPathDataflowPath &self) {
                     return aigLongestPathDataflowPathGetDelay(self);
                   })
      .def_prop_ro("fan_in",
                   [](AIGLongestPathDataflowPath &self) {
                     return aigLongestPathDataflowPathGetFanIn(self);
                   })
      .def_prop_ro("fan_out",
                   [](AIGLongestPathDataflowPath &self) {
                     return aigLongestPathDataflowPathGetFanOut(self);
                   })
      .def_prop_ro("history",
                   [](AIGLongestPathDataflowPath &self) {
                     return aigLongestPathDataflowPathGetHistory(self);
                   })
      .def_prop_ro("root", [](AIGLongestPathDataflowPath &self) {
        return aigLongestPathDataflowPathGetRoot(self);
      });

  nb::class_<AIGLongestPathHistory>(m, "_LongestPathHistory")
      .def_prop_ro("empty",
                   [](AIGLongestPathHistory &self) {
                     return aigLongestPathHistoryIsEmpty(self);
                   })
      .def_prop_ro("head",
                   [](AIGLongestPathHistory &self) {
                     AIGLongestPathObject object;
                     int64_t delay;
                     MlirStringRef comment;
                     aigLongestPathHistoryGetHead(self, &object, &delay,
                                                  &comment);
                     return std::make_tuple(object, delay, comment);
                   })
      .def_prop_ro("tail", [](AIGLongestPathHistory &self) {
        return aigLongestPathHistoryGetTail(self);
      });

  nb::class_<AIGLongestPathObject>(m, "_LongestPathObject")
      .def_prop_ro("instance_path",
                   [](AIGLongestPathObject &self) {
                     auto path = aigLongestPathObjectGetInstancePath(self);
                     if (!path.ptr)
                       return std::vector<MlirOperation>();
                     size_t size = igraphInstancePathSize(path);
                     std::vector<MlirOperation> result;
                     for (size_t i = 0; i < size; ++i)
                       result.push_back(igraphInstancePathGet(path, i));
                     return result;
                   })
      .def_prop_ro("name",
                   [](AIGLongestPathObject &self) {
                     return aigLongestPathObjectName(self);
                   })
      .def_prop_ro("bit_pos", [](AIGLongestPathObject &self) {
        return aigLongestPathObjectBitPos(self);
      });
}
