//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/FIRRTL/LayerSet.h"
#include "circt/Dialect/FIRRTL/FIRRTLOps.h"
#include "circt/Dialect/FIRRTL/FIRRTLUtils.h"
#include "llvm/ADT/SmallSet.h"

using namespace mlir;
using namespace circt;
using namespace firrtl;

LayerSet circt::firrtl::getAmbientLayersAt(Operation *op) {
  // Crawl through the parent ops, accumulating all ambient layers at the given
  // operation.
  LayerSet result;
  for (; op != nullptr; op = op->getParentOp()) {
    if (auto module = dyn_cast<FModuleLike>(op)) {
      auto layers = module.getLayersAttr().getAsRange<SymbolRefAttr>();
      result.insert(layers.begin(), layers.end());
      break;
    }
    if (auto layerblock = dyn_cast<LayerBlockOp>(op)) {
      result.insert(layerblock.getLayerName());
      continue;
    }
  }
  return result;
}

LayerSet circt::firrtl::getAmbientLayersFor(Value value) {
  return getAmbientLayersAt(getFieldRefFromValue(value).getDefiningOp());
}

LayerSet circt::firrtl::getLayersFor(Value value) {
  auto result = getAmbientLayersFor(value);
  if (auto type = dyn_cast<RefType>(value.getType()))
    if (auto layer = type.getLayer())
      result.insert(type.getLayer());
  return result;
}

bool circt::firrtl::isLayerCompatibleWith(mlir::SymbolRefAttr srcLayer,
                                          mlir::SymbolRefAttr dstLayer) {
  // A non-colored probe may be cast to any colored probe.
  if (!srcLayer)
    return true;

  // A colored probe cannot be cast to an uncolored probe.
  if (!dstLayer)
    return false;

  // Return true if the srcLayer is a prefix of the dstLayer.
  if (srcLayer.getRootReference() != dstLayer.getRootReference())
    return false;

  auto srcNames = srcLayer.getNestedReferences();
  auto dstNames = dstLayer.getNestedReferences();
  if (dstNames.size() < srcNames.size())
    return false;

  return llvm::all_of(llvm::zip_first(srcNames, dstNames),
                      [](auto x) { return std::get<0>(x) == std::get<1>(x); });
}

bool circt::firrtl::isLayerCompatibleWith(SymbolRefAttr srcLayer,
                                          const LayerSet &dstLayers) {
  // fast path: the required layer is directly listed in the provided layers.
  if (dstLayers.contains(srcLayer))
    return true;

  // Slow path: the required layer is not directly listed in the provided
  // layers, but the layer may still be provided by a nested layer.
  return any_of(dstLayers, [=](SymbolRefAttr dstLayer) {
    return isLayerCompatibleWith(srcLayer, dstLayer);
  });
}

bool circt::firrtl::isLayerSetCompatibleWith(
    const LayerSet &src, const LayerSet &dst,
    SmallVectorImpl<SymbolRefAttr> &missing) {
  for (auto srcLayer : src)
    if (!isLayerCompatibleWith(srcLayer, dst))
      missing.push_back(srcLayer);

  llvm::sort(missing, LayerSetCompare());
  return missing.empty();
}

LogicalResult circt::firrtl::checkLayerCompatibility(Operation *op,
                                                     const LayerSet &src,
                                                     const LayerSet &dst,
                                                     const Twine &errorMsg,
                                                     const Twine &noteMsg) {
  SmallVector<SymbolRefAttr> missing;
  if (isLayerSetCompatibleWith(src, dst, missing))
    return success();
  interleaveComma(missing, op->emitOpError(errorMsg).attachNote()
                               << noteMsg << ": ");
  return failure();
}
