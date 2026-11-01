//===- DebugInfoUtils.cpp - Helpers for debug info emission ---------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "DebugInfoUtils.h"
#include "circt/Analysis/DebugInfo.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "llvm/Support/FileSystem.h"

using namespace mlir;
using namespace circt;

/// Walk the given `loc` and collect file-line-column locations that we want to
/// report as source ("HGL") locations or as emitted Verilog ("HDL") locations.
///
/// This function treats locations inside a `NameLoc` called "emitted" or a
/// `FusedLoc` with the metadata attribute string "verilogLocations" as emitted
/// Verilog locations. All other locations are considered to be source
/// locations.
///
/// The `level` parameter is used to track into how many "emitted" or
/// "verilogLocations" we have already descended. For every one of those we look
/// through the level gets decreased by one. File-line-column locations are only
/// collected at level 0. We don't descend into "emitted" or "verilogLocations"
/// once we've reached level 0. This effectively makes the `level` parameter
/// decide behind how many layers of "emitted" or "verilogLocations" we want to
/// collect file-line-column locations. Setting this to 0 effectively collects
/// source locations, i.e., everything not marked as emitted. Setting this to 1
/// effectively collects emitted locations, i.e., nothing that isn't behind
/// exactly one layer of "emitted" or "verilogLocations".
static void findLocations(Location loc, unsigned level,
                          SmallVectorImpl<FileLineColLoc> &locs) {
  if (auto nameLoc = dyn_cast<NameLoc>(loc)) {
    if (nameLoc.getName() == "emitted")
      if (level-- == 0)
        return;
    findLocations(nameLoc.getChildLoc(), level, locs);
  } else if (auto fusedLoc = dyn_cast<FusedLoc>(loc)) {
    auto strAttr = dyn_cast_or_null<StringAttr>(fusedLoc.getMetadata());
    if (strAttr && strAttr.getValue() == "verilogLocations")
      if (level-- == 0)
        return;
    for (auto innerLoc : fusedLoc.getLocations())
      findLocations(innerLoc, level, locs);
  } else if (auto fileLoc = dyn_cast<FileLineColLoc>(loc)) {
    if (level == 0)
      locs.push_back(fileLoc);
  }
}

FileLineColLoc circt::findBestLocation(Location loc, bool emitted,
                                       bool fileMustExist) {
  SmallVector<FileLineColLoc> locs;
  findLocations(loc, emitted ? 1 : 0, locs);
  if (fileMustExist) {
    unsigned tail = 0;
    for (unsigned head = 0, end = locs.size(); head != end; ++head)
      if (llvm::sys::fs::exists(locs[head].getFilename().getValue()))
        locs[tail++] = locs[head];
    locs.resize(tail);
  }
  for (auto loc : locs)
    if (!loc.getFilename().getValue().ends_with(".fir"))
      return loc;
  for (auto loc : locs)
    if (loc.getFilename().getValue().ends_with(".fir"))
      return loc;
  return {};
}

StringAttr circt::getVerilogModuleName(DIModule &module) {
  if (auto *op = module.op)
    if (auto attr = op->getAttrOfType<StringAttr>("verilogName"))
      return attr;
  return module.name;
}

StringAttr circt::getVerilogInstanceName(DIInstance &inst) {
  if (auto *op = inst.op)
    if (auto attr = op->getAttrOfType<StringAttr>("hw.verilogName"))
      return attr;
  return inst.name;
}
