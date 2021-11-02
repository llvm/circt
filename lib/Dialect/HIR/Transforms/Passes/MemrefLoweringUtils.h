//=========- MemrefLoweringUtils.h - Utils for memref lowering pass---======//
//
// This header declares utilities for memref lowering pass.
//
//===----------------------------------------------------------------------===//

#include "../PassDetails.h"
#include <iostream>
using namespace circt;
using namespace hir;

Value insertDataSendLogic(OpBuilder &builder, Value data, Value inDataBus,
                          ArrayRef<Value> indices, Value tVar,
                          IntegerAttr offsetAttr);

Value insertDataRecvLogic(OpBuilder &builder, Location loc, ArrayAttr names,
                          Value dataBusT, ArrayRef<Value> indices,
                          uint64_t rdLatency, Value tVar,
                          IntegerAttr offsetAttr);

Value insertDataTensorSendLogic(OpBuilder &builder, Value currentDataBusT,
                                Value rootDataBusT, Value tVar,
                                IntegerAttr offsetAttr);
