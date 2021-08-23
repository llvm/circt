//===- ESI.cpp - C Interface for the ESI Dialect --------------------------===//
//
//===----------------------------------------------------------------------===//

#include "circt-c/Dialect/ESI.h"
#include "circt/Dialect/ESI/ESITypes.h"
#include "mlir/CAPI/IR.h"
#include "mlir/CAPI/Registration.h"
#include "mlir/CAPI/Support.h"
#include "mlir/CAPI/Utils.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Parser.h"
#include "mlir/Support/FileUtilities.h"

using namespace circt::esi;
using namespace mlir;

MLIR_DEFINE_CAPI_DIALECT_REGISTRATION(ESI, esi, circt::esi::ESIDialect)

void registerESIPasses() { circt::esi::registerESIPasses(); }

MlirLogicalResult circtESIExportCosimSchema(MlirModule module,
                                            MlirStringCallback callback,
                                            void *userData) {
  mlir::detail::CallbackOstream stream(callback, userData);
  return wrap(circt::esi::exportCosimSchema(unwrap(module), stream));
}

bool circtESITypeIsAChannelType(MlirType type) {
  return unwrap(type).isa<ChannelPort>();
}

MlirType circtESIChannelTypeGet(MlirType inner) {
  auto cppInner = unwrap(inner);
  return wrap(ChannelPort::get(cppInner.getContext(), cppInner));
}

MlirType circtESIChannelGetInner(MlirType channelType) {
  return wrap(unwrap(channelType).cast<ChannelPort>().getInner());
}

MlirOperation circtESIWrapModule(MlirOperation cModOp, long numPorts,
                                 const MlirStringRef *ports) {
  mlir::Operation *modOp = unwrap(cModOp);
  llvm::SmallVector<llvm::StringRef, 8> portNamesRefs;
  for (long i = 0; i < numPorts; ++i)
    portNamesRefs.push_back(ports[i].data);
  llvm::SmallVector<ESIPortValidReadyMapping, 8> portTriples;
  resolvePortNames(modOp, portNamesRefs, portTriples);
  mlir::OpBuilder b(modOp);
  mlir::Operation *wrapper = buildESIWrapper(b, modOp, portTriples);
  return wrap(wrapper);
}

void circtESIAppendMlirFile(MlirModule cMod, MlirStringRef filename) {
  ModuleOp modOp = unwrap(cMod);
  auto loadedMod = parseSourceFile(unwrap(filename), modOp.getContext());
  Block *loadedBlock = loadedMod->getBody();
  assert(!modOp->getRegions().empty());
  if (modOp.body().empty()) {
    modOp.body().push_back(loadedBlock);
    return;
  }
  auto &ops = modOp.getBody()->getOperations();
  ops.splice(ops.end(), loadedBlock->getOperations());
}
MlirOperation circtESILookup(MlirModule mod, MlirStringRef symbol) {
  return wrap(SymbolTable::lookupSymbolIn(unwrap(mod), unwrap(symbol)));
}
