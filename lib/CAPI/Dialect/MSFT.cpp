//===- MSFT.cpp - C Interface for the MSFT Dialect ------------------------===//
//
//===----------------------------------------------------------------------===//

#include "circt-c/Dialect/MSFT.h"
#include "circt/Dialect/MSFT/DeviceDB.h"
#include "circt/Dialect/MSFT/ExportTcl.h"
#include "circt/Dialect/MSFT/MSFTAttributes.h"
#include "circt/Dialect/MSFT/MSFTDialect.h"
#include "circt/Dialect/MSFT/MSFTOps.h"
#include "circt/Support/LLVM.h"
#include "mlir/CAPI/IR.h"
#include "mlir/CAPI/Registration.h"
#include "mlir/CAPI/Support.h"
#include "mlir/CAPI/Utils.h"
#include "mlir/Transforms/Passes.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/raw_ostream.h"

MLIR_DEFINE_CAPI_DIALECT_REGISTRATION(MSFT, msft, circt::msft::MSFTDialect)

using namespace circt;
using namespace circt::msft;

void mlirMSFTRegisterPasses() {
  mlir::registerCanonicalizerPass();
  circt::msft::registerMSFTPasses();
}

//===----------------------------------------------------------------------===//
// PrimitiveDB.
//===----------------------------------------------------------------------===//

DEFINE_C_API_PTR_METHODS(CirctMSFTPrimitiveDB, circt::msft::PrimitiveDB)

CirctMSFTPrimitiveDB circtMSFTCreatePrimitiveDB(MlirContext ctxt) {
  return wrap(new PrimitiveDB(unwrap(ctxt)));
}
void circtMSFTDeletePrimitiveDB(CirctMSFTPrimitiveDB self) {
  delete unwrap(self);
}
MlirLogicalResult circtMSFTPrimitiveDBAddPrimitive(CirctMSFTPrimitiveDB self,
                                                   MlirAttribute cLoc) {
  PhysLocationAttr loc = unwrap(cLoc).cast<PhysLocationAttr>();
  return wrap(unwrap(self)->addPrimitive(loc));
}
bool circtMSFTPrimitiveDBIsValidLocation(CirctMSFTPrimitiveDB self,
                                         MlirAttribute cLoc) {
  PhysLocationAttr loc = unwrap(cLoc).cast<PhysLocationAttr>();
  return unwrap(self)->isValidLocation(loc);
}

//===----------------------------------------------------------------------===//
// PlacementDB.
//===----------------------------------------------------------------------===//

DEFINE_C_API_PTR_METHODS(CirctMSFTPlacementDB, circt::msft::PlacementDB)

CirctMSFTPlacementDB circtMSFTCreatePlacementDB(MlirOperation top,
                                                CirctMSFTPrimitiveDB seed) {
  if (seed.ptr == nullptr)
    return wrap(new PlacementDB(unwrap(top)));
  return wrap(new PlacementDB(unwrap(top), *unwrap(seed)));
}
void circtMSFTDeletePlacementDB(CirctMSFTPlacementDB self) {
  delete unwrap(self);
}
size_t circtMSFTPlacementDBAddDesignPlacements(CirctMSFTPlacementDB self) {
  return unwrap(self)->addDesignPlacements();
}
MlirLogicalResult
circtMSFTPlacementDBAddPlacement(CirctMSFTPlacementDB self, MlirAttribute cLoc,
                                 CirctMSFTPlacedInstance cInst) {
  Attribute attr = unwrap(cLoc);
  ArrayAttr path = unwrap(cInst.path).cast<ArrayAttr>();
  StringAttr subpath = StringAttr::get(
      attr.getContext(), StringRef(cInst.subpath, cInst.subpathLength));
  auto inst =
      PlacementDB::PlacedInstance{path, subpath.getValue(), unwrap(cInst.op)};

  if (auto loc = attr.dyn_cast<PhysLocationAttr>())
    return wrap(unwrap(self)->addPlacement(loc, inst));
  if (auto regionRef = attr.dyn_cast<PhysicalRegionRefAttr>())
    return wrap(unwrap(self)->addPlacement(regionRef, inst));

  return wrap(failure());
}
MlirLogicalResult circtMSFTPlacementDBRemovePlacement(CirctMSFTPlacementDB db,
                                                      MlirAttribute cLoc) {
  auto loc = unwrap(cLoc).cast<PhysLocationAttr>();
  return wrap(unwrap(db)->removePlacement(loc));
}
MlirLogicalResult circtMSFTPlacementDBMovePlacement(CirctMSFTPlacementDB db,
                                                    MlirAttribute cOldLoc,
                                                    MlirAttribute cNewLoc) {
  auto oldLoc = unwrap(cOldLoc).cast<PhysLocationAttr>();
  auto newLoc = unwrap(cNewLoc).cast<PhysLocationAttr>();
  return wrap(unwrap(db)->movePlacement(oldLoc, newLoc));
}
bool circtMSFTPlacementDBTryGetInstanceAt(CirctMSFTPlacementDB self,
                                          MlirAttribute cLoc,
                                          CirctMSFTPlacedInstance *out) {
  auto loc = unwrap(cLoc).cast<PhysLocationAttr>();
  Optional<PlacementDB::PlacedInstance> inst = unwrap(self)->getInstanceAt(loc);
  if (!inst)
    return false;
  if (out != nullptr) {
    out->path = wrap(inst->path);
    out->subpath = inst->subpath.data();
    out->subpathLength = inst->subpath.size();
    out->op = wrap(inst->op);
  }
  return true;
}

MlirAttribute circtMSFTPlacementDBGetNearestFreeInColumn(
    CirctMSFTPlacementDB cdb, CirctMSFTPrimitiveType prim, uint64_t column,
    uint64_t nearestToY) {
  auto db = unwrap(cdb);
  return wrap(
      db->getNearestFreeInColumn((PrimitiveType)prim, column, nearestToY));
}

void circtMSFTPlacementDBWalkPlacements(CirctMSFTPlacementDB cdb,
                                        CirctMSFTPlacementCallback ccb,
                                        int64_t bounds[4],
                                        CirctMSFTPrimitiveType cPrimTypeFilter,
                                        CirctMSFTWalkOrder cWalkOrder,
                                        void *userData) {
  PlacementDB *db = unwrap(cdb);
  auto cb = [ccb, userData](PhysLocationAttr loc,
                            PlacementDB::PlacedInstance p) {
    CirctMSFTPlacedInstance cPlacement = {wrap(p.path), p.subpath.data(),
                                          p.subpath.size(), wrap(p.op)};
    ccb(wrap(loc), cPlacement, userData);
  };
  Optional<PrimitiveType> primTypeFilter;
  if (cPrimTypeFilter >= 0)
    primTypeFilter = static_cast<PrimitiveType>(cPrimTypeFilter);

  Optional<PlacementDB::WalkOrder> walkOrder;
  if (cWalkOrder.columns != CirctMSFTDirection::NONE ||
      cWalkOrder.rows != CirctMSFTDirection::NONE)
    walkOrder = PlacementDB::WalkOrder{
        static_cast<PlacementDB::Direction>(cWalkOrder.columns),
        static_cast<PlacementDB::Direction>(cWalkOrder.rows)};

  db->walkPlacements(
      cb, std::make_tuple(bounds[0], bounds[1], bounds[2], bounds[3]),
      primTypeFilter, walkOrder);
}

//===----------------------------------------------------------------------===//
// MSFT Attributes.
//===----------------------------------------------------------------------===//

void mlirMSFTAddPhysLocationAttr(MlirOperation cOp, const char *entityName,
                                 PrimitiveType type, long x, long y, long num) {
  mlir::Operation *op = unwrap(cOp);
  mlir::MLIRContext *ctxt = op->getContext();
  PhysLocationAttr loc = PhysLocationAttr::get(
      ctxt, PrimitiveTypeAttr::get(ctxt, type), x, y, num, entityName);
  llvm::SmallString<64> entity("loc:");
  entity.append(entityName);
  op->setAttr(entity, loc);
}

bool circtMSFTAttributeIsAPhysLocationAttribute(MlirAttribute attr) {
  return unwrap(attr).isa<PhysLocationAttr>();
}
MlirAttribute circtMSFTPhysLocationAttrGet(MlirContext cCtxt,
                                           CirctMSFTPrimitiveType devType,
                                           uint64_t x, uint64_t y, uint64_t num,
                                           MlirStringRef subPath) {
  auto *ctxt = unwrap(cCtxt);
  return wrap(PhysLocationAttr::get(
      ctxt, PrimitiveTypeAttr::get(ctxt, (PrimitiveType)devType), x, y, num,
      unwrap(subPath)));
}

CirctMSFTPrimitiveType
circtMSFTPhysLocationAttrGetPrimitiveType(MlirAttribute attr) {
  return (CirctMSFTPrimitiveType)unwrap(attr)
      .cast<PhysLocationAttr>()
      .getPrimitiveType()
      .getValue();
}
uint64_t circtMSFTPhysLocationAttrGetX(MlirAttribute attr) {
  return (CirctMSFTPrimitiveType)unwrap(attr).cast<PhysLocationAttr>().getX();
}
uint64_t circtMSFTPhysLocationAttrGetY(MlirAttribute attr) {
  return (CirctMSFTPrimitiveType)unwrap(attr).cast<PhysLocationAttr>().getY();
}
uint64_t circtMSFTPhysLocationAttrGetNum(MlirAttribute attr) {
  return (CirctMSFTPrimitiveType)unwrap(attr).cast<PhysLocationAttr>().getNum();
}

bool circtMSFTAttributeIsAPhysicalBoundsAttr(MlirAttribute attr) {
  return unwrap(attr).isa<PhysicalBoundsAttr>();
}

MlirAttribute circtMSFTPhysicalBoundsAttrGet(MlirContext cContext,
                                             uint64_t xMin, uint64_t xMax,
                                             uint64_t yMin, uint64_t yMax) {
  auto *context = unwrap(cContext);
  return wrap(PhysicalBoundsAttr::get(context, xMin, xMax, yMin, yMax));
}

bool circtMSFTAttributeIsAPhysicalRegionRefAttr(MlirAttribute attr) {
  return unwrap(attr).isa<PhysicalRegionRefAttr>();
}

MlirAttribute circtMSFTPhysicalRegionRefAttrGet(MlirContext cContext,
                                                MlirStringRef cName) {
  auto *context = unwrap(cContext);
  auto name = unwrap(cName);
  auto ref = FlatSymbolRefAttr::get(context, name);
  return wrap(PhysicalRegionRefAttr::get(context, ref));
}

MlirOperation circtMSFTGetInstance(MlirOperation cRoot, MlirAttribute cPath) {
  auto root = cast<circt::msft::MSFTModuleOp>(unwrap(cRoot));
  auto path = unwrap(cPath).cast<SymbolRefAttr>();
  return wrap(getInstance(root, path));
}
