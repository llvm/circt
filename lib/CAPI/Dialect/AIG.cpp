//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "circt-c/Dialect/AIG.h"
#include "circt/Dialect/AIG/AIGDialect.h"
#include "circt/Dialect/AIG/AIGPasses.h"
#include "circt/Dialect/AIG/Analysis/LongestPathAnalysis.h"
#include "mlir-c/BuiltinAttributes.h"
#include "mlir-c/IR.h"
#include "mlir-c/Support.h"
#include "mlir/CAPI/IR.h"
#include "mlir/CAPI/Registration.h"
#include "mlir/CAPI/Support.h"
#include "mlir/Pass/AnalysisManager.h"
#include "llvm/Support/JSON.h"

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

//===----------------------------------------------------------------------===//
// LongestPathAnalysis C API
//===----------------------------------------------------------------------===//

AIGLongestPathAnalysis aigLongestPathAnalysisCreate(MlirOperation module,
                                                    bool traceDebugPoints) {
  auto *op = unwrap(module);
  auto *wrapper = new LongestPathAnalysisWrapper();
  wrapper->analysisManager =
      std::make_unique<mlir::ModuleAnalysisManager>(op, nullptr);
  mlir::AnalysisManager am = *wrapper->analysisManager;
  if (traceDebugPoints)
    wrapper->analysis = std::make_unique<LongestPathAnalysisWithTrace>(op, am);
  else
    wrapper->analysis = std::make_unique<LongestPathAnalysis>(op, am);
  return wrap(wrapper);
}

void aigLongestPathAnalysisDestroy(AIGLongestPathAnalysis analysis) {
  delete unwrap(analysis);
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

MlirStringRef
aigLongestPathCollectionGetPath(AIGLongestPathCollection collection,
                                int pathIndex) {
  auto *wrapper = unwrap(collection);

  // Check if pathIndex is valid
  if (pathIndex < 0 || pathIndex >= static_cast<int>(wrapper->paths.size()))
    return wrap(llvm::StringRef(""));

  // Convert the specific path to JSON
  // FIXME: Avoid converting to JSON and then back to string. Use native
  // CAPI instead once data structure is stabilized.
  llvm::json::Value pathJson = toJSON(wrapper->paths[pathIndex]);

  std::string jsonStr;
  llvm::raw_string_ostream os(jsonStr);
  os << pathJson;

  auto ctx = wrap(wrapper->getContext());

  // Use MLIR StringAttr to manage the string lifetime.
  // FIXME: This is safe but expensive. Consider manually managing the string
  // lifetime.
  MlirAttribute strAttr =
      mlirStringAttrGet(ctx, mlirStringRefCreateFromCString(os.str().c_str()));
  return mlirStringAttrGetValue(strAttr);
}
