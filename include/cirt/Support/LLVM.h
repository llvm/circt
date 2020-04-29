//===- LLVM.h - Import and forward declare core LLVM types ------*- C++ -*-===//
//
// This file forward declares and imports various common LLVM and MLIR datatypes
// that we want to use unqualified.
//
// Note that most of these are forward declared and then imported into the cirt
// namespace with using decls, rather than being #included.  This is because we
// want clients to explicitly #include the files they need.
//
//===----------------------------------------------------------------------===//

#ifndef CIRT_SUPPORT_LLVM_H
#define CIRT_SUPPORT_LLVM_H

// MLIR includes a lot of forward declarations of LLVM types, use them.
#include "mlir/Support/LLVM.h"

// Import things we want into our namespace.
namespace cirt {
// Casting operators.
using llvm::cast;
using llvm::cast_or_null;
using llvm::dyn_cast;
using llvm::dyn_cast_or_null;
using llvm::isa;
using llvm::isa_and_nonnull;

// Containers.
using llvm::ArrayRef;
using llvm::DenseMapInfo;
template <typename KeyT, typename ValueT,
          typename KeyInfoT = DenseMapInfo<KeyT>,
          typename BucketT = llvm::detail::DenseMapPair<KeyT, ValueT>>
using DenseMap = llvm::DenseMap<KeyT, ValueT, KeyInfoT, BucketT>;
template <typename ValueT, typename ValueInfoT = DenseMapInfo<ValueT>>
using DenseSet = llvm::DenseSet<ValueT, ValueInfoT>;
template <typename Fn>
using function_ref = llvm::function_ref<Fn>;
using llvm::iterator_range;
using llvm::MutableArrayRef;
using llvm::None;
using llvm::Optional;
using llvm::PointerUnion;
using llvm::SmallPtrSet;
using llvm::SmallPtrSetImpl;
using llvm::SmallString;
using llvm::SmallVector;
using llvm::SmallVectorImpl;
using llvm::StringLiteral;
using llvm::StringRef;
using llvm::TinyPtrVector;
using llvm::Twine;

// Other common classes.
using llvm::APFloat;
using llvm::APInt;
using llvm::raw_ostream;
} // namespace cirt

#endif // CIRT_SUPPORT_LLVM_H
