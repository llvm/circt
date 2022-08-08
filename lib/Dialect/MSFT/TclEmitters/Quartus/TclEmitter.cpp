//===- TclEmitter.cpp - Emit Quartus-flavored, Stratix-10 targeted TCL ----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Write out Tcl with the appropriate API calls for a Stratix-10 device in
// Quartus-flavored Tcl.
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/MSFT/TclEmitter.h"
#include "circt/Dialect/HW/HWAttributes.h"
#include "circt/Dialect/HW/HWOps.h"
#include "circt/Dialect/HW/HWSymCache.h"
#include "circt/Dialect/HW/HWTypes.h"
#include "circt/Dialect/MSFT/DeviceDB.h"
#include "circt/Dialect/MSFT/MSFTAttributes.h"
#include "circt/Dialect/MSFT/MSFTOps.h"
#include "circt/Dialect/SV/SVOps.h"
#include "circt/Dialect/Seq/SeqOps.h"
#include "circt/Support/LLVM.h"
#include "mlir/IR/Builders.h"
#include "mlir/Tools/mlir-translate/Translation.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/raw_ostream.h"

#include "circt/Dialect/MSFT/TclEmitters/Quartus/TclEmitter.h"

using namespace circt;
using namespace hw;
using namespace msft;

void QuartusTclOutputState::emit(PhysLocationAttr pla) {
  // Different devices have different 'number' letters (the 'N' in 'N0'). M20Ks
  // and DSPs happen to have the same one, probably because they never co-exist
  // at the same location.
  char numCharacter;
  switch (pla.getPrimitiveType().getValue()) {
  case PrimitiveType::M20K:
    os << "M20K";
    numCharacter = 'N';
    break;
  case PrimitiveType::DSP:
    os << "MPDSP";
    numCharacter = 'N';
    break;
  case PrimitiveType::FF:
    os << "FF";
    numCharacter = 'N';
    break;
  }

  // Write out the rest of the location info.
  os << "_X" << pla.getX() << "_Y" << pla.getY() << "_" << numCharacter
     << pla.getNum();
}

/// Emit tcl in the form of:
/// "set_location_assignment MPDSP_X34_Y285_N0 -to
/// $parent|fooInst|entityName(subpath)"
LogicalResult
QuartusTclOutputState::emitLocationAssignment(DynInstDataOpInterface refOp,
                                              PhysLocationAttr loc,
                                              Optional<StringRef> subpath) {
  indent() << "set_location_assignment ";
  emit(loc);

  // To which entity does this apply?
  os << " -to $parent|";
  emitPath(getRefOp(refOp), subpath);

  return success();
}

LogicalResult QuartusTclOutputState::emit(PDPhysLocationOp loc) {
  if (failed(emitLocationAssignment(loc, loc.loc(), loc.subPath())))
    return failure();
  os << '\n';
  return success();
}

LogicalResult QuartusTclOutputState::emit(PDRegPhysLocationOp locs) {
  ArrayRef<PhysLocationAttr> locArr = locs.locs().getLocs();
  for (size_t i = 0, e = locArr.size(); i < e; ++i) {
    PhysLocationAttr pla = locArr[i];
    if (!pla)
      continue;
    if (failed(emitLocationAssignment(locs, pla, {})))
      return failure();
    os << "[" << i << "]\n";
  }
  return success();
}

/// Emit tcl in the form of:
/// "set_global_assignment -name NAME VALUE -to $parent|fooInst|entityName"
LogicalResult QuartusTclOutputState::emit(DynamicInstanceVerbatimAttrOp attr) {
  GlobalRefOp ref = getRefOp(attr);
  indent() << "set_instance_assignment -name " << attr.name() << " "
           << attr.value();

  // To which entity does this apply?
  os << " -to $parent|";
  emitPath(ref, attr.subPath());
  os << '\n';
  return success();
}

/// Emit tcl in the form of:
/// set_instance_assignment -name PLACE_REGION "X1 Y1 X20 Y20" -to $parent|a|b|c
/// set_instance_assignment -name RESERVE_PLACE_REGION OFF -to $parent|a|b|c
/// set_instance_assignment -name CORE_ONLY_PLACE_REGION ON -to $parent|a|b|c
/// set_instance_assignment -name REGION_NAME test_region -to $parent|a|b|c
LogicalResult QuartusTclOutputState::emit(PDPhysRegionOp region) {
  GlobalRefOp ref = getRefOp(region);

  auto physicalRegion = dyn_cast_or_null<DeclPhysicalRegionOp>(
      emitter.getDefinition(region.physRegionRefAttr()));
  if (!physicalRegion)
    return region.emitOpError(
               "could not find physical region declaration named ")
           << region.physRegionRefAttr();

  // PLACE_REGION directive.
  indent() << "set_instance_assignment -name PLACE_REGION \"";
  auto physicalBounds =
      physicalRegion.bounds().getAsRange<PhysicalBoundsAttr>();
  llvm::interleave(
      physicalBounds, os,
      [&](PhysicalBoundsAttr bounds) {
        os << 'X' << bounds.getXMin() << ' ';
        os << 'Y' << bounds.getYMin() << ' ';
        os << 'X' << bounds.getXMax() << ' ';
        os << 'Y' << bounds.getYMax();
      },
      ";");
  os << '"';

  os << " -to $parent|";
  emitPath(ref, region.subPath());
  os << '\n';

  // RESERVE_PLACE_REGION directive.
  indent() << "set_instance_assignment -name RESERVE_PLACE_REGION OFF";
  os << " -to $parent|";
  emitPath(ref, region.subPath());
  os << '\n';

  // CORE_ONLY_PLACE_REGION directive.
  indent() << "set_instance_assignment -name CORE_ONLY_PLACE_REGION ON";
  os << " -to $parent|";
  emitPath(ref, region.subPath());
  os << '\n';

  // REGION_NAME directive.
  indent() << "set_instance_assignment -name REGION_NAME ";
  os << physicalRegion.getName();
  os << " -to $parent|";
  emitPath(ref, region.subPath());
  os << '\n';
  return success();
}
