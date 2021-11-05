#include "../PassDetails.h"
using namespace circt;
using namespace hir;

Value hwSelect(OpBuilder &builder, Value sel, Value trueVal, Value falseVal);

std::pair<Value, Value> insertForOpEntryLogic(OpBuilder &builder,
                                              Value isFirstIter, Value lb,
                                              Value ub, Value step,
                                              Value tstart);
