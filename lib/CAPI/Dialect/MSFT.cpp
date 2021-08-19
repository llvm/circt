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

MlirLogicalResult mlirMSFTExportTcl(MlirModule module,
                                    MlirStringCallback callback,
                                    void *userData) {
  mlir::detail::CallbackOstream stream(callback, userData);
  return wrap(exportQuartusTcl(unwrap(module), stream));
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

bool circtMSFTAttributeIsASwitchInstanceAttribute(MlirAttribute attr) {
  return unwrap(attr).isa<SwitchInstanceAttr>();
}
MlirAttribute circtMSFTSwitchInstanceAttrGet(
    MlirContext cCtxt, CirctMSFTInstIDAttrPair *listOfCases, size_t numCases) {
  SmallVector<InstIDAttrPair, 64> cases;
  for (size_t i = 0; i < numCases; ++i) {
    CirctMSFTInstIDAttrPair pair = listOfCases[i];
    auto instance = unwrap(pair.instance).cast<SymbolRefAttr>();
    auto attr = unwrap(pair.attr);
    cases.push_back(std::make_pair(instance, attr));
  }
  return wrap(SwitchInstanceAttr::get(unwrap(cCtxt), cases));
}
size_t circtMSFTSwitchInstanceAttrGetNumCases(MlirAttribute attr) {
  return unwrap(attr).cast<SwitchInstanceAttr>().getCases().size();
}
void circtMSFTSwitchInstanceAttrGetCases(MlirAttribute attr,
                                         CirctMSFTInstIDAttrPair *dstArray,
                                         size_t space) {
  auto sw = unwrap(attr).cast<SwitchInstanceAttr>();
  ArrayRef<InstIDAttrPair> cases = sw.getCases();
  assert(space >= cases.size());
  for (size_t i = 0, e = cases.size(); i < e; ++i) {
    auto c = cases[i];
    dstArray[i] = {wrap(c.first), wrap(c.second)};
  }
}
