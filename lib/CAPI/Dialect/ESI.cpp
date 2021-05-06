//===- ESI.cpp - C Interface for the ESI Dialect --------------------------===//
//
//===----------------------------------------------------------------------===//

#include "circt-c/Dialect/ESI.h"
#include "circt/Dialect/ESI/ESITypes.h"
#include "mlir/CAPI/IR.h"
#include "mlir/CAPI/Registration.h"
#include "mlir/CAPI/Support.h"

using namespace circt::esi;

MLIR_DEFINE_CAPI_DIALECT_REGISTRATION(ESI, esi, circt::esi::ESIDialect)

void registerESIPasses() { circt::esi::registerESIPasses(); }

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
