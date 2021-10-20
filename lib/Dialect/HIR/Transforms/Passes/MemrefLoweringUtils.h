//=========- MemrefLoweringUtils.h - Utils for memref lowering pass---======//
//
// This header declares utilities for memref lowering pass.
//
//===----------------------------------------------------------------------===//

#include "../PassDetails.h"
#include <iostream>
using namespace circt;
using namespace hir;
void insertEnableSendLogic(OpBuilder &builder, Value inEnableBus,
                           Value outEnableBus, Value tVar,
                           IntegerAttr offsetAttr, ArrayRef<Value> indices);

void insertDataSendLogic(OpBuilder &builder, Value data, Value enableBus,
                         Value inDataBus, Value outDataBus, Value tVar,
                         IntegerAttr offsetAttr, ArrayRef<Value> indices);
