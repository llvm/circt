//===- MSFT.cpp - C Interface for the MSFT Dialect ------------------------===//
//
//===----------------------------------------------------------------------===//

#include "circt-c/Dialect/MSFT.h"
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

void mlirMSFTRegisterGenerator(MlirContext cCtxt, const char *opName,
                               const char *generatorName,
                               mlirMSFTGeneratorCallback cb,
                               MlirAttribute parameters) {
  mlir::MLIRContext *ctxt = unwrap(cCtxt);
  MSFTDialect *msft = ctxt->getLoadedDialect<MSFTDialect>();
  msft->registerGenerator(
      llvm::StringRef(opName), llvm::StringRef(generatorName),
      [cb](mlir::Operation *op) {
        return unwrap(cb.callback(wrap(op), cb.userData));
      },
      unwrap(parameters));
}

void mlirMSFTAddPhysLocationAttr(MlirOperation cOp, const char *entityName,
                                 DeviceType type, long x, long y, long num) {
  mlir::Operation *op = unwrap(cOp);
  mlir::MLIRContext *ctxt = op->getContext();
  PhysLocationAttr loc =
      PhysLocationAttr::get(ctxt, DeviceTypeAttr::get(ctxt, type), x, y, num);
  llvm::SmallString<64> entity("loc:");
  entity.append(entityName);
  op->setAttr(entity, loc);
}

bool circtMSFTAttributeIsAPhysLocationAttribute(MlirAttribute attr) {
  return unwrap(attr).isa<PhysLocationAttr>();
}
MlirAttribute circtMSFTPhysLocationAttrGet(MlirContext cCtxt,
                                           CirctMSFTDevType devType, uint64_t x,
                                           uint64_t y, uint64_t num) {
  auto ctxt = unwrap(cCtxt);
  return wrap(PhysLocationAttr::get(
      ctxt, DeviceTypeAttr::get(ctxt, (DeviceType)devType), x, y, num));
}

CirctMSFTDevType circtMSFTPhysLocationAttrGetDeviceType(MlirAttribute attr) {
  return (CirctMSFTDevType)unwrap(attr)
      .cast<PhysLocationAttr>()
      .getDevType()
      .getValue();
}
uint64_t circtMSFTPhysLocationAttrGetX(MlirAttribute attr) {
  return (CirctMSFTDevType)unwrap(attr).cast<PhysLocationAttr>().getX();
}
uint64_t circtMSFTPhysLocationAttrGetY(MlirAttribute attr) {
  return (CirctMSFTDevType)unwrap(attr).cast<PhysLocationAttr>().getY();
}
uint64_t circtMSFTPhysLocationAttrGetNum(MlirAttribute attr) {
  return (CirctMSFTDevType)unwrap(attr).cast<PhysLocationAttr>().getNum();
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
