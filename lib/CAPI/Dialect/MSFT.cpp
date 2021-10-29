//===- MSFT.cpp - C Interface for the MSFT Dialect ------------------------===//
//
//===----------------------------------------------------------------------===//

#include "circt-c/Dialect/MSFT.h"
#include "circt/Dialect/MSFT/DeviceDB.h"
#include "circt/Dialect/MSFT/ExportTcl.h"
#include "circt/Dialect/MSFT/MSFTAttributes.h"
#include "circt/Dialect/MSFT/MSFTDialect.h"
#include "circt/Support/LLVM.h"
#include "mlir/CAPI/IR.h"
#include "mlir/CAPI/Registration.h"
#include "mlir/CAPI/Support.h"
#include "mlir/CAPI/Utils.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/raw_ostream.h"

MLIR_DEFINE_CAPI_DIALECT_REGISTRATION(MSFT, msft, circt::msft::MSFTDialect)

using namespace circt;
using namespace circt::msft;

void mlirMSFTRegisterPasses() { circt::msft::registerMSFTPasses(); }

MlirLogicalResult mlirMSFTExportTcl(MlirOperation module,
                                    MlirStringCallback callback,
                                    void *userData) {
  Operation *op = unwrap(module);
  hw::HWModuleOp hwmod = dyn_cast<hw::HWModuleOp>(op);
  if (!hwmod)
    return wrap(op->emitOpError("Export TCL can only be run on HWModules"));
  mlir::detail::CallbackOstream stream(callback, userData);
  return wrap(exportQuartusTcl(hwmod, stream));
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
  RootedInstancePathAttr path =
      unwrap(cInst.path).cast<RootedInstancePathAttr>();
  StringAttr subpath = StringAttr::get(
      attr.getContext(), StringRef(cInst.subpath, cInst.subpathLength));
  auto inst =
      PlacementDB::PlacedInstance{path, subpath.getValue(), unwrap(cInst.op)};

  if (auto loc = attr.dyn_cast<PhysLocationAttr>())
    return wrap(unwrap(self)->addPlacement(loc, inst));
  if (auto region = attr.dyn_cast<LogicLockedRegionAttr>())
    return wrap(unwrap(self)->addPlacement(region, inst));

  return wrap(failure());
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
  db->walkPlacements(
      cb, std::make_tuple(bounds[0], bounds[1], bounds[2], bounds[3]),
      primTypeFilter);
}

//===----------------------------------------------------------------------===//
// MSFT Attributes.
//===----------------------------------------------------------------------===//

void mlirMSFTAddPhysLocationAttr(MlirOperation cOp, const char *entityName,
                                 PrimitiveType type, long x, long y, long num) {
  mlir::Operation *op = unwrap(cOp);
  mlir::MLIRContext *ctxt = op->getContext();
  PhysLocationAttr loc = PhysLocationAttr::get(
      ctxt, PrimitiveTypeAttr::get(ctxt, type), x, y, num);
  llvm::SmallString<64> entity("loc:");
  entity.append(entityName);
  op->setAttr(entity, loc);
}

bool circtMSFTAttributeIsAPhysLocationAttribute(MlirAttribute attr) {
  return unwrap(attr).isa<PhysLocationAttr>();
}
MlirAttribute circtMSFTPhysLocationAttrGet(MlirContext cCtxt,
                                           CirctMSFTPrimitiveType devType,
                                           uint64_t x, uint64_t y,
                                           uint64_t num) {
  auto ctxt = unwrap(cCtxt);
  return wrap(PhysLocationAttr::get(
      ctxt, PrimitiveTypeAttr::get(ctxt, (PrimitiveType)devType), x, y, num));
}

bool circtMSFTAttributeIsALogicLockedRegionAttr(MlirAttribute attr) {
  return unwrap(attr).isa<LogicLockedRegionAttr>();
}
MlirAttribute circtMSFTAttributeLogicLockedRegionAttrGet(
    MlirContext cCtxt, MlirStringRef regionName, uint64_t xMin, uint64_t xMax,
    uint64_t yMin, uint64_t yMax) {
  auto ctxt = unwrap(cCtxt);
  auto regionNameAttr = StringAttr::get(ctxt, unwrap(regionName));
  return wrap(
      LogicLockedRegionAttr::get(ctxt, regionNameAttr, xMin, xMax, yMin, yMax));
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

bool circtMSFTAttributeIsARootedInstancePathAttribute(MlirAttribute cAttr) {
  return unwrap(cAttr).isa<RootedInstancePathAttr>();
}
MlirAttribute
circtMSFTRootedInstancePathAttrGet(MlirContext cCtxt, MlirAttribute cRootSym,
                                   MlirAttribute *cPathStringAttrs,
                                   size_t num) {
  auto ctxt = unwrap(cCtxt);
  auto rootSym = unwrap(cRootSym).cast<FlatSymbolRefAttr>();
  SmallVector<StringAttr, 16> path;
  for (size_t i = 0; i < num; ++i)
    path.push_back(unwrap(cPathStringAttrs[i]).cast<StringAttr>());
  return wrap(RootedInstancePathAttr::get(ctxt, rootSym, path));
}

bool circtMSFTAttributeIsASwitchInstanceAttribute(MlirAttribute attr) {
  return unwrap(attr).isa<SwitchInstanceAttr>();
}
MlirAttribute
circtMSFTSwitchInstanceAttrGet(MlirContext cCtxt,
                               CirctMSFTSwitchInstanceCase *listOfCases,
                               size_t numCases) {
  MLIRContext *ctxt = unwrap(cCtxt);
  SmallVector<SwitchInstanceCaseAttr, 64> cases;
  for (size_t i = 0; i < numCases; ++i) {
    CirctMSFTSwitchInstanceCase pair = listOfCases[i];
    Attribute instanceAttr = unwrap(pair.instance);
    auto instance = instanceAttr.dyn_cast<RootedInstancePathAttr>();
    assert(instance &&
           "Expected `RootedInstancePathAttr` in switch instance case.");
    auto attr = unwrap(pair.attr);
    cases.push_back(SwitchInstanceCaseAttr::get(ctxt, instance, attr));
  }
  return wrap(SwitchInstanceAttr::get(ctxt, cases));
}
size_t circtMSFTSwitchInstanceAttrGetNumCases(MlirAttribute attr) {
  return unwrap(attr).cast<SwitchInstanceAttr>().getCases().size();
}
void circtMSFTSwitchInstanceAttrGetCases(MlirAttribute attr,
                                         CirctMSFTSwitchInstanceCase *dstArray,
                                         size_t space) {
  auto sw = unwrap(attr).cast<SwitchInstanceAttr>();
  ArrayRef<SwitchInstanceCaseAttr> cases = sw.getCases();
  assert(space >= cases.size());
  for (size_t i = 0, e = cases.size(); i < e; ++i) {
    auto c = cases[i];
    dstArray[i] = {wrap(c.getInst()), wrap(c.getAttr())};
  }
}

MlirOperation circtMSFTGetInstance(MlirOperation cRoot, MlirAttribute cPath) {
  auto root = cast<circt::hw::HWModuleOp>(unwrap(cRoot));
  auto path = unwrap(cPath).cast<SymbolRefAttr>();
  return wrap(getInstance(root, path));
}
