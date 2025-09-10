//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "circt-c/Dialect/AIG.h"
#include "circt-c/Support/InstanceGraph.h"
#include "circt/Dialect/AIG/AIGDialect.h"
#include "circt/Dialect/AIG/AIGPasses.h"
#include "circt/Dialect/AIG/Analysis/LongestPathAnalysis.h"
#include "circt/Support/InstanceGraph.h"
#include "circt/Support/InstanceGraphInterface.h"
#include "mlir-c/BuiltinAttributes.h"
#include "mlir-c/IR.h"
#include "mlir-c/Support.h"
#include "mlir/CAPI/IR.h"
#include "mlir/CAPI/Registration.h"
#include "mlir/CAPI/Support.h"
#include "mlir/Pass/AnalysisManager.h"
#include "llvm/ADT/ImmutableList.h"
#include "llvm/ADT/PointerUnion.h"
#include "llvm/Support/JSON.h"
#include <memory>
#include <tuple>

using namespace circt;
using namespace circt::aig;

MLIR_DEFINE_CAPI_DIALECT_REGISTRATION(AIG, aig, circt::aig::AIGDialect)

void registerAIGPasses() { circt::aig::registerPasses(); }

// Wrapper struct to hold both the analysis and the analysis manager
struct LongestPathAnalysisWrapper {
  std::unique_ptr<mlir::ModuleAnalysisManager> analysisManager;
  std::unique_ptr<LongestPathAnalysis> analysis;
};

DEFINE_C_API_PTR_METHODS(AIGLongestPathAnalysis, LongestPathAnalysisWrapper)
DEFINE_C_API_PTR_METHODS(AIGLongestPathCollection, LongestPathCollection)
DEFINE_C_API_PTR_METHODS(AIGLongestPathDataflowPath, DataflowPath)
DEFINE_C_API_PTR_METHODS(AIGLongestPathHistory,
                         llvm::ImmutableListImpl<DebugPoint>)

// AIGLongestPathObject is a pointer to either an Object or an OutputPort so we
// can not use DEFINE_C_API_PTR_METHODS.
llvm::PointerUnion<Object *, DataflowPath::OutputPort *>
unwrap(AIGLongestPathObject object) {
  return llvm::PointerUnion<
      Object *, DataflowPath::OutputPort *>::getFromOpaqueValue(object.ptr);
}

AIGLongestPathObject
wrap(llvm::PointerUnion<Object *, DataflowPath::OutputPort *> object) {
  return AIGLongestPathObject{object.getOpaqueValue()};
}

AIGLongestPathObject wrap(const Object *object) {
  auto ptr = llvm::PointerUnion<Object *, DataflowPath::OutputPort *>(
      const_cast<Object *>(object));
  return wrap(ptr);
}

AIGLongestPathObject wrap(const DataflowPath::OutputPort *object) {
  auto ptr = llvm::PointerUnion<Object *, DataflowPath::OutputPort *>(
      const_cast<DataflowPath::OutputPort *>(object));
  return wrap(ptr);
}

//===----------------------------------------------------------------------===//
// LongestPathAnalysis C API
//===----------------------------------------------------------------------===//

AIGLongestPathAnalysis aigLongestPathAnalysisCreate(MlirOperation module,
                                                    bool collectDebugInfo,
                                                    bool keepOnlyMaxDelayPaths,
                                                    bool lazyComputation) {
  auto *op = unwrap(module);
  auto *wrapper = new LongestPathAnalysisWrapper();
  wrapper->analysisManager =
      std::make_unique<mlir::ModuleAnalysisManager>(op, nullptr);
  mlir::AnalysisManager am = *wrapper->analysisManager;
  wrapper->analysis = std::make_unique<LongestPathAnalysis>(
      op, am,
      LongestPathAnalysisOption(collectDebugInfo, lazyComputation,
                                keepOnlyMaxDelayPaths));
  return wrap(wrapper);
}

void aigLongestPathAnalysisDestroy(AIGLongestPathAnalysis analysis) {
  delete unwrap(analysis);
}

AIGLongestPathCollection
aigLongestPathAnalysisGetPaths(AIGLongestPathAnalysis analysis, MlirValue value,
                               int64_t bitPos, bool elaboratePaths) {
  auto *wrapper = unwrap(analysis);
  auto *lpa = wrapper->analysis.get();
  auto *collection = new LongestPathCollection(lpa->getContext());
  auto result =
      lpa->computeGlobalPaths(unwrap(value), bitPos, collection->paths);
  if (failed(result))
    return {nullptr};
  collection->sortInDescendingOrder();
  return wrap(collection);
}

AIGLongestPathCollection
aigLongestPathAnalysisGetAllPaths(AIGLongestPathAnalysis analysis,
                                  MlirStringRef moduleName,
                                  bool elaboratePaths) {
  auto *wrapper = unwrap(analysis);
  auto *lpa = wrapper->analysis.get();
  auto moduleNameAttr = StringAttr::get(lpa->getContext(), unwrap(moduleName));

  auto *collection = new LongestPathCollection(lpa->getContext());
  if (!lpa->isAnalysisAvailable(moduleNameAttr) ||
      failed(
          lpa->getAllPaths(moduleNameAttr, collection->paths, elaboratePaths)))
    return {nullptr};

  collection->sortInDescendingOrder();
  return wrap(collection);
}

// ===----------------------------------------------------------------------===//
// LongestPathCollection
// ===----------------------------------------------------------------------===//

bool aigLongestPathCollectionIsNull(AIGLongestPathCollection collection) {
  return !collection.ptr;
}

void aigLongestPathCollectionDestroy(AIGLongestPathCollection collection) {
  delete unwrap(collection);
}

size_t aigLongestPathCollectionGetSize(AIGLongestPathCollection collection) {
  auto *wrapper = unwrap(collection);
  return wrapper->paths.size();
}

// Get a specific path from the collection as DataflowPath object
AIGLongestPathDataflowPath
aigLongestPathCollectionGetDataflowPath(AIGLongestPathCollection collection,
                                        size_t index) {
  auto *wrapper = unwrap(collection);
  auto &path = wrapper->paths[index];
  return wrap(&path);
}

void aigLongestPathCollectionMerge(AIGLongestPathCollection dest,
                                   AIGLongestPathCollection src) {
  auto *destWrapper = unwrap(dest);
  auto *srcWrapper = unwrap(src);
  destWrapper->merge(*srcWrapper);
}

//===----------------------------------------------------------------------===//
// DataflowPath
//===----------------------------------------------------------------------===//

int64_t aigLongestPathDataflowPathGetDelay(AIGLongestPathDataflowPath path) {
  auto *wrapper = unwrap(path);
  return wrapper->getDelay();
}

AIGLongestPathObject
aigLongestPathDataflowPathGetFanIn(AIGLongestPathDataflowPath path) {
  auto *wrapper = unwrap(path);
  auto &fanIn = wrapper->getFanIn();
  return wrap(const_cast<Object *>(&fanIn));
}

AIGLongestPathObject
aigLongestPathDataflowPathGetFanOut(AIGLongestPathDataflowPath path) {
  auto *wrapper = unwrap(path);
  if (auto *object = std::get_if<Object>(&wrapper->getFanOut())) {
    return wrap(object);
  }
  auto *ptr = std::get_if<DataflowPath::OutputPort>(&wrapper->getFanOut());
  return wrap(ptr);
}

AIGLongestPathHistory
aigLongestPathDataflowPathGetHistory(AIGLongestPathDataflowPath path) {
  auto *wrapper = unwrap(path);
  return wrap(const_cast<llvm::ImmutableListImpl<DebugPoint> *>(
      wrapper->getHistory().getInternalPointer()));
}

MlirOperation
aigLongestPathDataflowPathGetRoot(AIGLongestPathDataflowPath path) {
  auto *wrapper = unwrap(path);
  return wrap(wrapper->getRoot());
}

//===----------------------------------------------------------------------===//
// History
//===----------------------------------------------------------------------===//

bool aigLongestPathHistoryIsEmpty(AIGLongestPathHistory history) {
  auto *wrapper = unwrap(history);
  return llvm::ImmutableList<DebugPoint>(wrapper).isEmpty();
}

void aigLongestPathHistoryGetHead(AIGLongestPathHistory history,
                                  AIGLongestPathObject *object, int64_t *delay,
                                  MlirStringRef *comment) {
  auto *wrapper = unwrap(history);
  auto list = llvm::ImmutableList<DebugPoint>(wrapper);

  auto &head = list.getHead();
  *object = wrap(&head.object);
  *delay = head.delay;
  *comment = mlirStringRefCreate(head.comment.data(), head.comment.size());
}

AIGLongestPathHistory
aigLongestPathHistoryGetTail(AIGLongestPathHistory history) {
  auto *wrapper = unwrap(history);
  auto list = llvm::ImmutableList<DebugPoint>(wrapper);
  auto *tail = list.getTail().getInternalPointer();
  return wrap(const_cast<llvm::ImmutableListImpl<DebugPoint> *>(tail));
}

//===----------------------------------------------------------------------===//
// Object
//===----------------------------------------------------------------------===//

IgraphInstancePath
aigLongestPathObjectGetInstancePath(AIGLongestPathObject object) {
  auto *ptr = dyn_cast<Object *>(unwrap(object));
  if (ptr) {
    IgraphInstancePath result;
    result.ptr = const_cast<igraph::InstanceOpInterface *>(
        ptr->instancePath.getPath().data());
    result.size = ptr->instancePath.getPath().size();
    return result;
  }

  // This is output port so the instance path is empty.
  return {nullptr, 0};
}

MlirStringRef aigLongestPathObjectName(AIGLongestPathObject rawObject) {
  auto ptr = unwrap(rawObject);
  if (auto *object = dyn_cast<Object *>(ptr)) {
    auto name = object->getName();
    return mlirStringRefCreate(name.data(), name.size());
  }
  auto [module, resultNumber, _] = *dyn_cast<DataflowPath::OutputPort *>(ptr);
  auto name = module.getOutputName(resultNumber);
  return mlirStringRefCreate(name.data(), name.size());
}

size_t aigLongestPathObjectBitPos(AIGLongestPathObject rawObject) {
  auto ptr = unwrap(rawObject);
  if (auto *object = dyn_cast<Object *>(ptr))
    return object->bitPos;
  return std::get<2>(*dyn_cast<DataflowPath::OutputPort *>(ptr));
}
