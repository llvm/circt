//=========- MemrefLoweringUtils.h - Utils for memref lowering pass---======//
//
// This header declares utilities for memref lowering pass.
//
//===----------------------------------------------------------------------===//

#include "../PassDetails.h"
#include <iostream>
using namespace circt;
using namespace hir;
Value insertEnableSendLogic(OpBuilder &builder, Value inEnableBus, Value tVar,
                            IntegerAttr offsetAttr, ArrayRef<Value> indices);

Value insertDataSendLogic(OpBuilder &builder, Value data, Value inDataBus,
                          Value tVar, IntegerAttr offsetAttr,
                          ArrayRef<Value> indices);

Value insertDataRecvLogic(OpBuilder &builder, Location loc, ArrayAttr names,
                          uint64_t rdLatency, Value dataBusT, Value tVar,
                          IntegerAttr offsetAttr, ArrayRef<Value> indices);
