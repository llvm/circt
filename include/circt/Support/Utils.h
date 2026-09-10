//===- Utils.h - Miscellaneous utilities ----------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Miscellaneous utilities for CIRCT that do not fit in with other files.
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_SUPPORT_UTILS_H
#define CIRCT_SUPPORT_UTILS_H

#include "circt/Support/LLVM.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Value.h"
#include "llvm/Support/Parallel.h"

namespace circt {

/// Return true if a Value is created "underneath" an operation.  This is
/// frequently useful when negated as that indicates that a Value is defined
/// "outside" the region of an Operation and that the Operation is thereby
/// "capturing" the value.
inline bool isAncestorOfValueOwner(Operation *op, Value value) {
  return op->isAncestor(value.getParentBlock()->getParentOp());
}

/// Remove elements from the input array corresponding to set bits in
/// `indicesToDrop`, returning the elements not mentioned.
template <typename T>
static SmallVector<T>
removeElementsAtIndices(ArrayRef<T> input,
                        const llvm::BitVector &indicesToDrop) {
#ifndef NDEBUG
  if (!input.empty()) {
    int lastIndex = indicesToDrop.find_last();
    if (lastIndex >= 0)
      assert((size_t)lastIndex < input.size() && "index out of range");
  }
#endif

  // If the input is empty (which is an optimization we do for certain array
  // attributes), simply return an empty vector.
  if (input.empty())
    return {};

  // Copy over the live chunks.
  size_t lastCopied = 0;
  SmallVector<T> result;
  result.reserve(input.size() - indicesToDrop.count());

  for (unsigned indexToDrop : indicesToDrop.set_bits()) {
    // If we skipped over some valid elements, copy them over.
    if (indexToDrop > lastCopied) {
      result.append(input.begin() + lastCopied, input.begin() + indexToDrop);
      lastCopied = indexToDrop;
    }
    // Ignore this value so we don't copy it in the next iteration.
    ++lastCopied;
  }

  // If there are live elements at the end, copy them over.
  if (lastCopied < input.size())
    result.append(input.begin() + lastCopied, input.end());

  return result;
}

//===----------------------------------------------------------------------===//
// Parallel utilities
//===----------------------------------------------------------------------===//

/// Wrapper for llvm::parallelTransformReduce that performs the transform_reduce
/// serially when MLIR multi-threading is disabled.
/// Does not add a ParallelDiagnosticHandler like mlir::parallelFor.
template <class IterTy, class ResultTy, class ReduceFuncTy,
          class TransformFuncTy>
static ResultTy transformReduce(MLIRContext *context, IterTy begin, IterTy end,
                                ResultTy init, ReduceFuncTy reduce,
                                TransformFuncTy transform) {
  // Parallel when enabled
  if (context->isMultithreadingEnabled())
    return llvm::parallelTransformReduce(begin, end, init, reduce, transform);

  // Serial fallback (from llvm::parallelTransformReduce)
  for (IterTy i = begin; i != end; ++i)
    init = reduce(std::move(init), transform(*i));
  return std::move(init);
}

/// Range wrapper
template <class RangeTy, class ResultTy, class ReduceFuncTy,
          class TransformFuncTy>
static ResultTy transformReduce(MLIRContext *context, RangeTy &&r,
                                ResultTy init, ReduceFuncTy reduce,
                                TransformFuncTy transform) {
  return transformReduce(context, std::begin(r), std::end(r), init, reduce,
                         transform);
}

} // namespace circt

#endif // CIRCT_SUPPORT_UTILS_H
