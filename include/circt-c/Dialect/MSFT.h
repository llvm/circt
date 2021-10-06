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
typedef uint32_t CirctMSFTPrimitiveType;

/// Emits tcl for the specified module using the provided callback and user
/// data
MLIR_CAPI_EXPORTED MlirLogicalResult mlirMSFTExportTcl(MlirOperation,
                                                       MlirStringCallback,
                                                       void *userData);

//===----------------------------------------------------------------------===//
// DeviceDB.
//===----------------------------------------------------------------------===//

typedef struct {
  void *ptr;
} CirctMSFTDeviceDB;

CirctMSFTDeviceDB circtMSFTCreateDeviceDB(MlirContext);
void circtMSFTDeleteDeviceDB(CirctMSFTDeviceDB self);
MlirLogicalResult circtMSFTDeviceDBAddPrimitive(CirctMSFTDeviceDB,
                                                MlirAttribute locAndPrim);
bool circtMSFTDeviceDBIsValidLocation(CirctMSFTDeviceDB,
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

CirctMSFTPlacementDB circtMSFTCreatePlacementDB(MlirOperation top,
                                                CirctMSFTDeviceDB seed);
void circtMSFTDeletePlacementDB(CirctMSFTPlacementDB self);
size_t circtMSFTPlacementDBAddDesignPlacements(CirctMSFTPlacementDB);
MlirLogicalResult
circtMSFTPlacementDBAddPlacement(CirctMSFTPlacementDB, MlirAttribute loc,
                                 CirctMSFTPlacedInstance inst);
bool circtMSFTPlacementDBTryGetInstanceAt(CirctMSFTPlacementDB,
                                          MlirAttribute loc,
                                          CirctMSFTPlacedInstance *out);
MlirAttribute circtMSFTPlacementDBGetNearestFreeInColumn(
    CirctMSFTPlacementDB, CirctMSFTPrimitiveType prim, uint64_t column,
    uint64_t nearestToY);

//===----------------------------------------------------------------------===//
// MSFT Attributes.
//===----------------------------------------------------------------------===//

/// Add a physical location attribute with the given entity name, device type, x
/// and y coordinates, and number.
MLIR_CAPI_EXPORTED void mlirMSFTAddPhysLocationAttr(MlirOperation op,
                                                    const char *entityName,
                                                    PrimitiveType type, long x,
                                                    long y, long num);

bool circtMSFTAttributeIsAPhysLocationAttribute(MlirAttribute);
MlirAttribute circtMSFTPhysLocationAttrGet(MlirContext, CirctMSFTPrimitiveType,
                                           uint64_t x, uint64_t y,
                                           uint64_t num);
CirctMSFTPrimitiveType circtMSFTPhysLocationAttrGetPrimitiveType(MlirAttribute);
uint64_t circtMSFTPhysLocationAttrGetX(MlirAttribute);
uint64_t circtMSFTPhysLocationAttrGetY(MlirAttribute);
uint64_t circtMSFTPhysLocationAttrGetNum(MlirAttribute);

bool circtMSFTAttributeIsARootedInstancePathAttribute(MlirAttribute);
MlirAttribute circtMSFTRootedInstancePathAttrGet(MlirContext,
                                                 MlirAttribute rootSym,
                                                 MlirAttribute *pathStringAttrs,
                                                 size_t num);

typedef struct {
  MlirAttribute instance;
  MlirAttribute attr;
} CirctMSFTSwitchInstanceCase;

bool circtMSFTAttributeIsASwitchInstanceAttribute(MlirAttribute);
MlirAttribute circtMSFTSwitchInstanceAttrGet(
    MlirContext, CirctMSFTSwitchInstanceCase *listOfCases, size_t numCases);
size_t circtMSFTSwitchInstanceAttrGetNumCases(MlirAttribute);
void circtMSFTSwitchInstanceAttrGetCases(MlirAttribute,
                                         CirctMSFTSwitchInstanceCase *dstArray,
                                         size_t space);

MlirOperation circtMSFTGetInstance(MlirOperation root, MlirAttribute path);

#ifdef __cplusplus
}
#endif

#endif // CIRCT_C_DIALECT_MSFT_H
