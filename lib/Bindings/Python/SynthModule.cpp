//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "CIRCTModules.h"

#include "circt-c/Dialect/Synth.h"

#include "mlir-c/Support.h"
#include "mlir/Bindings/Python/NanobindAdaptors.h"
#include "mlir/CAPI/Wrap.h"

#include "NanobindUtils.h"
#include <nanobind/nanobind.h>
#include <string_view>

namespace nb = nanobind;

using namespace circt;
using namespace mlir::python::nanobind_adaptors;

/// Populate the synth python module.
void circt::python::populateDialectSynthSubmodule(nb::module_ &m) {
  m.doc() = "Synth dialect Python native extension";

  // LongestPathAnalysis class
  nb::class_<SynthLongestPathAnalysis>(m, "_LongestPathAnalysis")
      .def(
          "__init__",
          [](SynthLongestPathAnalysis *self, MlirOperation module,
             bool collectDebugInfo, bool keepOnlyMaxDelayPaths,
             bool lazyComputation) {
            new (self) SynthLongestPathAnalysis(synthLongestPathAnalysisCreate(
                module, collectDebugInfo, keepOnlyMaxDelayPaths,
                lazyComputation));
          },
          nb::arg("module"), nb::arg("collect_debug_info") = false,
          nb::arg("keep_only_max_delay_paths") = false,
          nb::arg("lazy_computation") = false)
      .def("__del__",
           [](SynthLongestPathAnalysis &self) {
             synthLongestPathAnalysisDestroy(self);
           })
      .def("get_paths",
           [](SynthLongestPathAnalysis *self, MlirValue value, int64_t bitPos,
              bool elaboratePaths) -> SynthLongestPathCollection {
             auto collection =
                 SynthLongestPathCollection(synthLongestPathAnalysisGetPaths(
                     *self, value, bitPos, elaboratePaths));

             if (synthLongestPathCollectionIsNull(collection))
               throw nb::value_error(
                   "Failed to get all paths, see previous error(s).");

             return collection;
           })
      .def("get_all_paths",
           [](SynthLongestPathAnalysis *self, const std::string &moduleName,
              bool elaboratePaths) -> SynthLongestPathCollection {
             MlirStringRef moduleNameRef =
                 mlirStringRefCreateFromCString(moduleName.c_str());

             auto collection =
                 SynthLongestPathCollection(synthLongestPathAnalysisGetAllPaths(
                     *self, moduleNameRef, elaboratePaths));

             if (synthLongestPathCollectionIsNull(collection))
               throw nb::value_error(
                   "Failed to get all paths, see previous error(s).");

             return collection;
           });

  nb::class_<SynthLongestPathCollection>(m, "_LongestPathCollection")
      .def("__del__",
           [](SynthLongestPathCollection &self) {
             synthLongestPathCollectionDestroy(self);
           })
      .def("get_size",
           [](SynthLongestPathCollection &self) {
             return synthLongestPathCollectionGetSize(self);
           })
      .def("get_path",
           [](SynthLongestPathCollection &self,
              int pathIndex) -> SynthLongestPathDataflowPath {
             return synthLongestPathCollectionGetDataflowPath(self, pathIndex);
           })
      .def("merge", [](SynthLongestPathCollection &self,
                       SynthLongestPathCollection &src) {
        synthLongestPathCollectionMerge(self, src);
      });

  nb::class_<SynthLongestPathDataflowPath>(m, "_LongestPathDataflowPath")
      .def_prop_ro("delay",
                   [](SynthLongestPathDataflowPath &self) {
                     return synthLongestPathDataflowPathGetDelay(self);
                   })
      .def_prop_ro("fan_in",
                   [](SynthLongestPathDataflowPath &self) {
                     return synthLongestPathDataflowPathGetFanIn(self);
                   })
      .def_prop_ro("fan_out",
                   [](SynthLongestPathDataflowPath &self) {
                     return synthLongestPathDataflowPathGetFanOut(self);
                   })
      .def_prop_ro("history",
                   [](SynthLongestPathDataflowPath &self) {
                     return synthLongestPathDataflowPathGetHistory(self);
                   })
      .def_prop_ro("root", [](SynthLongestPathDataflowPath &self) {
        return synthLongestPathDataflowPathGetRoot(self);
      });

  nb::class_<SynthLongestPathHistory>(m, "_LongestPathHistory")
      .def_prop_ro("empty",
                   [](SynthLongestPathHistory &self) {
                     return synthLongestPathHistoryIsEmpty(self);
                   })
      .def_prop_ro("head",
                   [](SynthLongestPathHistory &self) {
                     SynthLongestPathObject object;
                     int64_t delay;
                     MlirStringRef comment;
                     synthLongestPathHistoryGetHead(self, &object, &delay,
                                                    &comment);
                     return std::make_tuple(object, delay, comment);
                   })
      .def_prop_ro("tail", [](SynthLongestPathHistory &self) {
        return synthLongestPathHistoryGetTail(self);
      });

  nb::class_<SynthLongestPathObject>(m, "_LongestPathObject")
      .def_prop_ro("instance_path",
                   [](SynthLongestPathObject &self) {
                     auto path = synthLongestPathObjectGetInstancePath(self);
                     if (!path.ptr)
                       return std::vector<MlirOperation>();
                     size_t size = igraphInstancePathSize(path);
                     std::vector<MlirOperation> result;
                     for (size_t i = 0; i < size; ++i)
                       result.push_back(igraphInstancePathGet(path, i));
                     return result;
                   })
      .def_prop_ro("name",
                   [](SynthLongestPathObject &self) {
                     return synthLongestPathObjectName(self);
                   })
      .def_prop_ro("bit_pos", [](SynthLongestPathObject &self) {
        return synthLongestPathObjectBitPos(self);
      });
}
