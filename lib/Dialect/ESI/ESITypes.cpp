//===- ESITypes.cpp - ESI types code defs -----------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Definitions for esi data types. Anything which doesn't have to be public
// should go in here.
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/ESI/ESITypes.h"
#include "circt/Dialect/ESI/ESIOps.h"
#include "circt/Dialect/HW/HWTypes.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/DialectImplementation.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace circt;
using namespace circt::esi;

AnyType AnyType::get(MLIRContext *context) { return Base::get(context); }

LogicalResult
WindowFieldType::verify(llvm::function_ref<InFlightDiagnostic()> emitError,
                        StringAttr fieldName, uint64_t numItems,
                        uint64_t bulkCountWidth) {
  if (numItems > 0 && bulkCountWidth > 0)
    return emitError() << "cannot specify both numItems and countWidth for "
                          "field '"
                       << fieldName.getValue() << "'";
  return success();
}

static unsigned getSignalingBitWidth(ChannelSignaling signaling) {
  switch (signaling) {
  case ChannelSignaling::ValidReady:
    return 2;
  case ChannelSignaling::FIFO:
    return 2;
  }
  llvm_unreachable("Unhandled ChannelSignaling");
}
std::optional<int64_t> ChannelType::getBitWidth() const {
  int64_t innerWidth = circt::hw::getBitWidth(getInner());
  if (innerWidth < 0)
    return std::nullopt;
  return innerWidth + getSignalingBitWidth(getSignaling());
}

/// Get the list of users with snoops filtered out. Returns a filtered range
/// which is lazily constructed.
static auto getChannelConsumers(mlir::TypedValue<ChannelType> chan) {
  return llvm::make_filter_range(chan.getUses(), [](auto &use) {
    return !isa<SnoopValidReadyOp, SnoopTransactionOp>(use.getOwner());
  });
}
SmallVector<std::reference_wrapper<OpOperand>, 4>
ChannelType::getConsumers(mlir::TypedValue<ChannelType> chan) {
  return SmallVector<std::reference_wrapper<OpOperand>, 4>(
      getChannelConsumers(chan));
}
bool ChannelType::hasOneConsumer(mlir::TypedValue<ChannelType> chan) {
  auto consumers = getChannelConsumers(chan);
  if (consumers.empty())
    return false;
  return ++consumers.begin() == consumers.end();
}
bool ChannelType::hasNoConsumers(mlir::TypedValue<ChannelType> chan) {
  return getChannelConsumers(chan).empty();
}
OpOperand *ChannelType::getSingleConsumer(mlir::TypedValue<ChannelType> chan) {
  auto consumers = getChannelConsumers(chan);
  auto iter = consumers.begin();
  if (iter == consumers.end())
    return nullptr;
  OpOperand *result = &*iter;
  if (++iter != consumers.end())
    return nullptr;
  return result;
}
LogicalResult ChannelType::verifyChannel(mlir::TypedValue<ChannelType> chan) {
  auto consumers = getChannelConsumers(chan);
  if (consumers.empty() || ++consumers.begin() == consumers.end())
    return success();
  auto err = chan.getDefiningOp()->emitOpError(
      "channels must have at most one consumer");
  for (auto &consumer : consumers)
    err.attachNote(consumer.getOwner()->getLoc()) << "channel used here";
  return err;
}

std::optional<int64_t> WindowType::getBitWidth() const {
  return hw::getBitWidth(getLoweredType());
}

LogicalResult
WindowType::verify(llvm::function_ref<InFlightDiagnostic()> emitError,
                   StringAttr name, Type into,
                   ArrayRef<WindowFrameType> frames) {
  auto structInto = hw::type_dyn_cast<hw::StructType>(into);
  if (!structInto)
    return emitError() << "only windows into structs are currently supported";

  // Build a map from field name to type for quick lookup.
  SmallDenseMap<StringAttr, Type> fieldTypes;
  for (hw::StructType::FieldInfo field : structInto.getElements())
    fieldTypes[field.name] = field.type;

  // Track which fields have been consumed (for non-bulk-transfer uses).
  DenseSet<StringAttr> consumedFields;

  for (auto frame : frames) {
    bool encounteredArrayOrListWithNumItems = false;

    for (WindowFieldType field : frame.getMembers()) {
      auto fieldTypeIter = fieldTypes.find(field.getFieldName());
      if (fieldTypeIter == fieldTypes.end())
        return emitError() << "invalid field name: " << field.getFieldName();

      Type fieldType = fieldTypeIter->getSecond();

      // Check if this field was already consumed by a previous frame.
      // Exception: bulk transfer mode (countWidth > 0) allows the same list
      // field to appear in a header frame (with countWidth) and later in a
      // data frame (with numItems).
      bool isBulkTransferHeader = field.getBulkCountWidth() > 0;
      if (consumedFields.contains(field.getFieldName()) &&
          !isBulkTransferHeader)
        return emitError() << "field '" << field.getFieldName()
                           << "' already consumed by a previous frame";

      // If we encounter an array or list field with numItems, no subsequent
      // fields in this frame can be arrays or lists with numItems.
      bool isArrayOrListWithNumItems =
          hw::type_isa<hw::ArrayType, esi::ListType>(fieldType) &&
          field.getNumItems() > 0;
      if (isArrayOrListWithNumItems) {
        if (encounteredArrayOrListWithNumItems)
          return emitError()
                 << "cannot have two array or list fields with num items (in "
                 << field.getFieldName() << ")";
        encounteredArrayOrListWithNumItems = true;
      }

      // If 'numItems' is specified, gotta run more checks.
      uint64_t numItems = field.getNumItems();
      if (numItems > 0) {
        if (auto arrField = hw::type_dyn_cast<hw::ArrayType>(fieldType)) {
          if (numItems > arrField.getNumElements())
            return emitError()
                   << "num items is larger than array size in field "
                   << field.getFieldName();
        } else if (!hw::type_isa<esi::ListType>(fieldType)) {
          return emitError() << "specification of num items only allowed on "
                                "array or list fields (in "
                             << field.getFieldName() << ")";
        }
      }

      // Bulk transfer mode validation: bulkCountWidth is only valid on list
      // fields.
      uint64_t bulkCountWidth = field.getBulkCountWidth();
      if (bulkCountWidth > 0) {
        if (!hw::type_isa<esi::ListType>(fieldType))
          return emitError() << "bulk transfer (countWidth) only allowed on "
                                "list fields (in "
                             << field.getFieldName() << ")";
      }

      // Mark this field as consumed (unless it's a bulk transfer header entry,
      // which allows the field to be used again in a data frame).
      if (!isBulkTransferHeader)
        consumedFields.insert(field.getFieldName());
    }
  }
  return success();
}

Type WindowType::getLoweredType() const {
  // Assemble a fast lookup of struct fields to types.
  auto into = hw::type_cast<hw::StructType>(getInto());
  SmallDenseMap<StringAttr, Type> intoFields;
  for (hw::StructType::FieldInfo field : into.getElements())
    intoFields[field.name] = field.type;

  auto getInnerTypeOrSelf = [&](Type t) {
    return TypeSwitch<Type, Type>(t)
        .Case<hw::ArrayType>(
            [](hw::ArrayType arr) { return arr.getElementType(); })
        .Case<esi::ListType>(
            [](esi::ListType list) { return list.getElementType(); })
        .Default([&](Type t) { return t; });
  };

  // Helper to wrap a lowered type in a TypeAlias if the 'into' type is a
  // TypeAlias.
  auto wrapInTypeAliasIfNeeded = [&](Type loweredType) -> Type {
    if (auto intoAlias = dyn_cast<hw::TypeAliasType>(getInto())) {
      auto intoRef = intoAlias.getRef();
      std::string aliasName = (Twine(intoRef.getLeafReference().getValue()) +
                               "_" + getName().getValue())
                                  .str();
      auto newRef = SymbolRefAttr::get(
          intoRef.getRootReference(),
          {FlatSymbolRefAttr::get(StringAttr::get(getContext(), aliasName))});
      return hw::TypeAliasType::get(newRef, loweredType);
    }
    return loweredType;
  };

  // First pass: identify which list fields have bulk transfer headers.
  // These fields should not have '_size' or 'last' in their data frames.
  DenseSet<StringAttr> bulkTransferFields;
  for (WindowFrameType frame : getFrames()) {
    for (WindowFieldType field : frame.getMembers()) {
      if (field.getBulkCountWidth() > 0)
        bulkTransferFields.insert(field.getFieldName());
    }
  }

  // Build the union, frame by frame
  SmallVector<hw::UnionType::FieldInfo, 4> unionFields;
  for (WindowFrameType frame : getFrames()) {

    // ... field by field.
    SmallVector<hw::StructType::FieldInfo, 4> fields;
    SmallVector<hw::StructType::FieldInfo, 4> leftOverFields;
    bool hasLeftOver = false;
    StringAttr leftOverName;

    for (WindowFieldType field : frame.getMembers()) {
      auto fieldTypeIter = intoFields.find(field.getFieldName());
      assert(fieldTypeIter != intoFields.end());
      auto fieldType = fieldTypeIter->getSecond();

      // Check for bulk transfer mode (countWidth specified on a list field).
      uint64_t bulkCountWidth = field.getBulkCountWidth();
      if (bulkCountWidth > 0) {
        // Bulk transfer header: add a 'fieldname_count' field with the
        // specified width. This count indicates how many items will be
        // transmitted in subsequent data frames.
        fields.push_back(
            {StringAttr::get(getContext(),
                             Twine(field.getFieldName().getValue()) + "_count"),
             IntegerType::get(getContext(), bulkCountWidth)});
        // Don't add any data or 'last' fields for the header entry.
        continue;
      }

      // Check if this field is part of a bulk transfer (has a header with
      // countWidth elsewhere).
      bool isBulkTransferData =
          bulkTransferFields.contains(field.getFieldName());

      // If the number of items isn't specified, just use the type.
      if (field.getNumItems() == 0) {
        // Directly use the type from the struct unless it's an array or list,
        // in which case we want the inner type.
        auto type = getInnerTypeOrSelf(fieldType);
        fields.push_back({field.getFieldName(), type});
        leftOverFields.push_back({field.getFieldName(), type});

        if (hw::type_isa<esi::ListType>(fieldType) && !isBulkTransferData) {
          // Lists need a 'last' signal to indicate the end of the list
          // (only in streaming mode, not bulk transfer).
          auto lastType = IntegerType::get(getContext(), 1);
          auto lastField = StringAttr::get(getContext(), "last");
          fields.push_back({lastField, lastType});
          leftOverFields.push_back({lastField, lastType});
        }
      } else {
        if (auto array =
                hw::type_dyn_cast<hw::ArrayType>(fieldTypeIter->getSecond())) {
          // The first union entry should be an array of length numItems.
          fields.push_back(
              {field.getFieldName(), hw::ArrayType::get(array.getElementType(),
                                                        field.getNumItems())});

          // If the array size is not a multiple of numItems, we need another
          // frame for the left overs.
          size_t leftOver = array.getNumElements() % field.getNumItems();
          if (leftOver) {
            // The verifier checks that there is only one field per frame with
            // numItems > 0.
            assert(!hasLeftOver);
            hasLeftOver = true;
            leftOverFields.push_back(
                {field.getFieldName(),
                 hw::ArrayType::get(array.getElementType(), leftOver)});

            leftOverName = StringAttr::get(
                getContext(), Twine(frame.getName().getValue(), "_leftOver"));
          }
        } else if (auto list = hw::type_cast<esi::ListType>(
                       fieldTypeIter->getSecond())) {
          // Add array of length numItems.
          fields.push_back(
              {field.getFieldName(),
               hw::ArrayType::get(list.getElementType(), field.getNumItems())});

          if (!isBulkTransferData) {
            // Streaming mode: add _size and last fields.
            // _size is clog2(numItems) in width.
            fields.push_back(
                {StringAttr::get(
                     getContext(),
                     Twine(field.getFieldName().getValue(), "_size")),
                 IntegerType::get(getContext(),
                                  llvm::Log2_64_Ceil(field.getNumItems()))});
            // Lists need a 'last' signal to indicate the end of the list.
            fields.push_back({StringAttr::get(getContext(), "last"),
                              IntegerType::get(getContext(), 1)});
          }
        } else {
          llvm_unreachable("numItems specified on non-array/list field");
        }
      }
    }

    // Special case: if we have one data frame and it doesn't have a name, don't
    // use a union.
    if (getFrames().size() == 1 && frame.getName().getValue().empty() &&
        !hasLeftOver) {
      auto loweredStruct = hw::StructType::get(getContext(), fields);
      return wrapInTypeAliasIfNeeded(loweredStruct);
    }

    if (!fields.empty())
      unionFields.push_back(
          {frame.getName(), hw::StructType::get(getContext(), fields), 0});

    if (hasLeftOver)
      unionFields.push_back(
          {leftOverName, hw::StructType::get(getContext(), leftOverFields), 0});
  }

  auto unionType = hw::UnionType::get(getContext(), unionFields);
  return wrapInTypeAliasIfNeeded(unionType);
}

namespace mlir {
template <>
struct FieldParser<::BundledChannel, ::BundledChannel> {
  static FailureOr<::BundledChannel> parse(AsmParser &p) {
    ChannelType type;
    std::string name;
    if (p.parseType(type))
      return failure();
    auto dir = FieldParser<::ChannelDirection>::parse(p);
    if (failed(dir))
      return failure();
    if (p.parseKeywordOrString(&name))
      return failure();
    return BundledChannel{StringAttr::get(p.getContext(), name), *dir, type};
  }
};
} // namespace mlir

namespace llvm {
inline ::llvm::raw_ostream &operator<<(::llvm::raw_ostream &p,
                                       ::BundledChannel channel) {
  p << channel.type << " " << channel.direction << " " << channel.name;
  return p;
}
} // namespace llvm

ChannelBundleType ChannelBundleType::getReversed() const {
  SmallVector<BundledChannel, 4> reversed;
  for (auto channel : getChannels())
    reversed.push_back({channel.name, flip(channel.direction), channel.type});
  return ChannelBundleType::get(getContext(), reversed, getResettable());
}

std::optional<int64_t> ChannelBundleType::getBitWidth() const {
  int64_t totalWidth = 0;
  for (auto channel : getChannels()) {
    std::optional<int64_t> channelWidth = channel.type.getBitWidth();
    if (!channelWidth)
      return std::nullopt;
    totalWidth += *channelWidth;
  }
  return totalWidth;
}

#define GET_TYPEDEF_CLASSES
#include "circt/Dialect/ESI/ESITypes.cpp.inc"

void ESIDialect::registerTypes() {
  addTypes<
#define GET_TYPEDEF_LIST
#include "circt/Dialect/ESI/ESITypes.cpp.inc"
      >();
}

mlir::Type circt::esi::innerType(mlir::Type type) {
  circt::esi::ChannelType chan =
      dyn_cast_or_null<circt::esi::ChannelType>(type);
  if (chan) // Unwrap the channel if it's a channel.
    type = chan.getInner();

  return type;
}
