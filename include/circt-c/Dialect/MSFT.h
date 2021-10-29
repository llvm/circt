//===-- circt-c/Dialect/MSFT.h - C API for MSFT dialect -----------*- C -*-===//
//
// This header declares the C interface for registering and accessing the
// MSFT dialect. A dialect should be registered with a context to make it
// available to users of the context. These users must load the dialect
// before using any of its attributes, operations or types. Parser and pass
// manager can load registered dialects automatically.
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_C_DIALECT_MSFT_H
#define CIRCT_C_DIALECT_MSFT_H

#include "circt/Dialect/MSFT/MSFTDialect.h"
#include "mlir-c/IR.h"
#include "mlir-c/Pass.h"
#include "mlir-c/Registration.h"

#ifdef __cplusplus
extern "C" {
#endif

MLIR_DECLARE_CAPI_DIALECT_REGISTRATION(MSFT, msft);

MLIR_CAPI_EXPORTED void mlirMSFTRegisterPasses();

// Values represented in `MSFT.td`.
typedef int32_t CirctMSFTPrimitiveType;

/// Emits tcl for the specified module using the provided callback and user
/// data
MLIR_CAPI_EXPORTED MlirLogicalResult mlirMSFTExportTcl(MlirOperation,
                                                       MlirStringCallback,
                                                       void *userData);

//===----------------------------------------------------------------------===//
// MSFT Attributes.
//===----------------------------------------------------------------------===//

/// Add a physical location attribute with the given entity name, device type, x
/// and y coordinates, and number.
MLIR_CAPI_EXPORTED void mlirMSFTAddPhysLocationAttr(MlirOperation op,
                                                    const char *entityName,
                                                    CirctMSFTPrimitiveType type,
                                                    long x, long y, long num);

MLIR_CAPI_EXPORTED bool
    circtMSFTAttributeIsAPhysLocationAttribute(MlirAttribute);
MLIR_CAPI_EXPORTED MlirAttribute circtMSFTPhysLocationAttrGet(
    MlirContext, CirctMSFTPrimitiveType, uint64_t x, uint64_t y, uint64_t num);
MLIR_CAPI_EXPORTED CirctMSFTPrimitiveType
    circtMSFTPhysLocationAttrGetPrimitiveType(MlirAttribute);
MLIR_CAPI_EXPORTED uint64_t circtMSFTPhysLocationAttrGetX(MlirAttribute);
MLIR_CAPI_EXPORTED uint64_t circtMSFTPhysLocationAttrGetY(MlirAttribute);
MLIR_CAPI_EXPORTED uint64_t circtMSFTPhysLocationAttrGetNum(MlirAttribute);

MLIR_CAPI_EXPORTED bool
    circtMSFTAttributeIsALogicLockedRegionAttr(MlirAttribute);
MLIR_CAPI_EXPORTED MlirAttribute circtMSFTAttributeLogicLockedRegionAttrGet(
    MlirContext, MlirStringRef regionName, uint64_t xMin, uint64_t xMax,
    uint64_t yMin, uint64_t yMax);

MLIR_CAPI_EXPORTED bool
    circtMSFTAttributeIsARootedInstancePathAttribute(MlirAttribute);
MLIR_CAPI_EXPORTED MlirAttribute
circtMSFTRootedInstancePathAttrGet(MlirContext, MlirAttribute rootSym,
                                   MlirAttribute *pathStringAttrs, size_t num);

typedef struct {
  MlirAttribute instance;
  MlirAttribute attr;
} CirctMSFTSwitchInstanceCase;

MLIR_CAPI_EXPORTED bool
    circtMSFTAttributeIsASwitchInstanceAttribute(MlirAttribute);
MLIR_CAPI_EXPORTED MlirAttribute circtMSFTSwitchInstanceAttrGet(
    MlirContext, CirctMSFTSwitchInstanceCase *listOfCases, size_t numCases);
MLIR_CAPI_EXPORTED size_t circtMSFTSwitchInstanceAttrGetNumCases(MlirAttribute);
MLIR_CAPI_EXPORTED void circtMSFTSwitchInstanceAttrGetCases(
    MlirAttribute, CirctMSFTSwitchInstanceCase *dstArray, size_t space);

MLIR_CAPI_EXPORTED MlirOperation circtMSFTGetInstance(MlirOperation root,
                                                      MlirAttribute path);

//===----------------------------------------------------------------------===//
// PrimitiveDB.
//===----------------------------------------------------------------------===//

typedef struct {
  void *ptr;
} CirctMSFTPrimitiveDB;

MLIR_CAPI_EXPORTED CirctMSFTPrimitiveDB circtMSFTCreatePrimitiveDB(MlirContext);
MLIR_CAPI_EXPORTED void circtMSFTDeletePrimitiveDB(CirctMSFTPrimitiveDB self);
MLIR_CAPI_EXPORTED MlirLogicalResult circtMSFTPrimitiveDBAddPrimitive(
    CirctMSFTPrimitiveDB, MlirAttribute locAndPrim);
MLIR_CAPI_EXPORTED bool
circtMSFTPrimitiveDBIsValidLocation(CirctMSFTPrimitiveDB,
                                    MlirAttribute locAndPrim);

//===----------------------------------------------------------------------===//
// PlacementDB.
//===----------------------------------------------------------------------===//

typedef struct {
  void *ptr;
} CirctMSFTPlacementDB;

typedef struct {
  MlirAttribute path; // RootedInstancePathAttr.
  const char *subpath;
  size_t subpathLength;
  MlirOperation op;
} CirctMSFTPlacedInstance;

MLIR_CAPI_EXPORTED CirctMSFTPlacementDB
circtMSFTCreatePlacementDB(MlirOperation top, CirctMSFTPrimitiveDB seed);
MLIR_CAPI_EXPORTED void circtMSFTDeletePlacementDB(CirctMSFTPlacementDB self);
MLIR_CAPI_EXPORTED
size_t circtMSFTPlacementDBAddDesignPlacements(CirctMSFTPlacementDB);
MLIR_CAPI_EXPORTED MlirLogicalResult circtMSFTPlacementDBAddPlacement(
    CirctMSFTPlacementDB, MlirAttribute loc, CirctMSFTPlacedInstance inst);
MLIR_CAPI_EXPORTED bool
circtMSFTPlacementDBTryGetInstanceAt(CirctMSFTPlacementDB, MlirAttribute loc,
                                     CirctMSFTPlacedInstance *out);
MLIR_CAPI_EXPORTED MlirAttribute circtMSFTPlacementDBGetNearestFreeInColumn(
    CirctMSFTPlacementDB, CirctMSFTPrimitiveType prim, uint64_t column,
    uint64_t nearestToY);

typedef void (*CirctMSFTPlacementCallback)(MlirAttribute loc,
                                           CirctMSFTPlacedInstance,
                                           void *userData);
/// Walk all the placements within 'bounds' ([xmin, xmax, ymin, ymax], inclusive
/// on all sides), with -1 meaning unbounded.
MLIR_CAPI_EXPORTED void circtMSFTPlacementDBWalkPlacements(
    CirctMSFTPlacementDB, CirctMSFTPlacementCallback, int64_t bounds[4],
    CirctMSFTPrimitiveType primTypeFilter, void *userData);

#ifdef __cplusplus
}
#endif

#endif // CIRCT_C_DIALECT_MSFT_H
