//===- CEmitterUtils.cpp - C emission utilities ---------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains functions for emitting C types from MLIR types.
// Most of these functions are copied from TranslateToCpp, which (unfortunately)
// are not publicly available.
//
//===----------------------------------------------------------------------===//

#include "circt/Tools/hlt/WrapGen/CEmitterUtils.h"

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/Operation.h"
#include "mlir/Support/LogicalResult.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/FormatVariadic.h"

using namespace mlir;

namespace circt {
namespace hlt {

/// Convenience functions to produce interleaved output with functions returning
/// a LogicalResult. This is different than those in STLExtras as functions used
/// on each element doesn't return a string.
template <typename ForwardIterator, typename UnaryFunctor,
          typename NullaryFunctor>
inline LogicalResult
interleaveWithError(ForwardIterator begin, ForwardIterator end,
                    UnaryFunctor eachFn, NullaryFunctor betweenFn) {
  if (begin == end)
    return success();
  if (failed(eachFn(*begin)))
    return failure();
  ++begin;
  for (; begin != end; ++begin) {
    betweenFn();
    if (failed(eachFn(*begin)))
      return failure();
  }
  return success();
}

template <typename Container, typename UnaryFunctor, typename NullaryFunctor>
inline LogicalResult interleaveWithError(const Container &c,
                                         UnaryFunctor eachFn,
                                         NullaryFunctor betweenFn) {
  return interleaveWithError(c.begin(), c.end(), eachFn, betweenFn);
}

template <typename Container, typename UnaryFunctor>
inline LogicalResult interleaveCommaWithError(const Container &c,
                                              raw_ostream &os,
                                              UnaryFunctor eachFn) {
  return interleaveWithError(c.begin(), c.end(), eachFn, [&]() { os << ", "; });
}

static bool shouldMapToUnsigned(IntegerType::SignednessSemantics val) {
  switch (val) {
  case IntegerType::Signless:
    return false;
  case IntegerType::Signed:
    return false;
  case IntegerType::Unsigned:
    return true;
  }
  llvm_unreachable("Unexpected IntegerType::SignednessSemantics");
}

LogicalResult emitType(llvm::raw_ostream &os, Location loc, Type type,
                       Optional<StringRef> variable) {
  if (auto memRefType = type.dyn_cast<MemRefType>()) {
    // We here follow the LLVM memref calling convention:
    // https://mlir.llvm.org/docs/TargetLLVMIR/#calling-conventions
    // where a memref is lowered into a set of arguments being:
    // %arg0 : memref<4xi32> ->
    // %arg0: !llvm.ptr<f32>   // Allocated pointer.
    // %arg1: !llvm.ptr<f32>   // Aligned pointer.
    // %arg2: i64              // Offset.
    // %arg3: i64              // Size in dim 0.
    // %arg4: i64              // Stride in dim 0.
    assert(memRefType.getRank() == 1 &&
           "Only unidimensional memories supported.. for now");

    // Allocated pointer. If we've been provided with a variable name, this wil
    // be the named variable.
    if (emitType(os, loc, memRefType.getElementType()).failed())
      return failure();
    os << "* ";
    if (variable)
      os << *variable;
    os << ", ";

    // Aligned pointer
    if (emitType(os, loc, memRefType.getElementType()).failed())
      return failure();
    os << "* ";
    if (variable)
      os << *variable << "_aligned_ptr";
    os << ", ";

    // Offset, size and stride
    if (variable) {
      os << "int64_t " << *variable << "_offset, ";
      os << "int64_t " << *variable << "_size, ";
      os << "int64_t " << *variable << "_stride";
    } else
      os << "int64_t, int64_t, int64_t";
    return success();
  } else if (auto iType = type.dyn_cast<IntegerType>()) {
    switch (iType.getWidth()) {
    case 1:
      (os << "bool");
      break;
    case 8:
    case 16:
    case 32:
    case 64: {
      if (shouldMapToUnsigned(iType.getSignedness()))
        (os << "uint" << iType.getWidth() << "_t");
      else
        (os << "int" << iType.getWidth() << "_t");
      break;
    }
    default:
      return emitError(loc, "cannot emit integer type ") << type;
    }
  } else if (auto fType = type.dyn_cast<FloatType>()) {
    switch (fType.getWidth()) {
    case 32:
      (os << "float");
      break;
    case 64:
      (os << "double");
      break;
    default:
      return emitError(loc, "cannot emit float type ") << type;
    }
  } else if (auto iType = type.dyn_cast<IndexType>())
    (os << "size_t");
  else if (auto tType = type.dyn_cast<TensorType>()) {
    if (!tType.hasRank())
      return emitError(loc, "cannot emit unranked tensor type");
    if (!tType.hasStaticShape())
      return emitError(loc, "cannot emit tensor type with non static shape");
    os << "Tensor<";
    if (failed(emitType(os, loc, tType.getElementType())))
      return failure();
    auto shape = tType.getShape();
    for (auto dimSize : shape) {
      os << ", ";
      os << dimSize;
    }
    os << ">";
    return success();
  } else if (auto tType = type.dyn_cast<TupleType>())
    return emitTupleType(os, loc, tType.getTypes());
  else {
    return emitError(loc, "cannot emit type ") << type;
  }

  if (variable)
    os << " " << *variable;
  return success();
}

LogicalResult emitTupleType(llvm::raw_ostream &os, Location loc,
                            TypeRange types) {
  os << "std::tuple<";
  if (failed(interleaveCommaWithError(
          types, os, [&](Type type) { return emitType(os, loc, type); })))
    return failure();
  os << ">";
  return success();
}

LogicalResult emitTypes(llvm::raw_ostream &os, Location loc, TypeRange types) {
  switch (types.size()) {
  case 0:
    os << "void";
    return success();
  case 1:
    return emitType(os, loc, types.front());
  default:
    return emitTupleType(os, loc, types);
  }
}

} // namespace hlt
} // namespace circt
