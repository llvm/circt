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

LogicalResult
WindowType::verify(llvm::function_ref<InFlightDiagnostic()> emitError,
                   StringAttr name, Type into,
                   ArrayRef<WindowFrameType> frames) {
  auto structInto = hw::type_dyn_cast<hw::StructType>(into);
  if (!structInto)
    return emitError() << "only windows into structs are currently supported";

  auto fields = structInto.getElements();
  for (auto frame : frames) {
    // Efficiently look up fields in the frame.
    DenseMap<StringAttr, WindowFieldType> frameFields;
    for (auto field : frame.getMembers())
      frameFields[field.getFieldName()] = field;

    // Iterate through the list of struct fields until we've encountered all the
    // fields listed in the frame.
    while (!fields.empty() && !frameFields.empty()) {
      hw::StructType::FieldInfo field = fields.front();
      fields = fields.drop_front();
      auto f = frameFields.find(field.name);

      // If a field in the struct isn't listed, it's being omitted from the
      // window so we just skip it.
      if (f == frameFields.end())
        continue;

      // If 'numItems' is specified, gotta run more checks.
      uint64_t numItems = f->getSecond().getNumItems();
      if (numItems > 0) {
        auto arrField = hw::type_dyn_cast<hw::ArrayType>(field.type);
        if (!arrField)
          return emitError() << "cannot specify num items on non-array field "
                             << field.name;
        if (numItems > arrField.getNumElements())
          return emitError() << "num items is larger than array size in field "
                             << field.name;
        if (frame.getMembers().size() != 1)
          return emitError()
                 << "array with size specified must be in their own frame (in "
                 << field.name << ")";
      }
      frameFields.erase(f);
    }

    // If there is anything left in the frame list, it either refers to a
    // non-existant field or said frame was already consumed by a previous
    // frame.
    if (!frameFields.empty())
      return emitError() << "invalid field name: "
                         << frameFields.begin()->getSecond().getFieldName();
  }
  return success();
}

hw::UnionType WindowType::getLoweredType() const {
  // Assemble a fast lookup of struct fields to types.
  auto into = hw::type_cast<hw::StructType>(getInto());
  SmallDenseMap<StringAttr, Type> intoFields;
  for (hw::StructType::FieldInfo field : into.getElements())
    intoFields[field.name] = field.type;

  // Build the union, frame by frame
  SmallVector<hw::UnionType::FieldInfo, 4> unionFields;
  for (WindowFrameType frame : getFrames()) {

    // ... field by field.
    SmallVector<hw::StructType::FieldInfo, 4> fields;
    for (WindowFieldType field : frame.getMembers()) {
      auto fieldTypeIter = intoFields.find(field.getFieldName());
      assert(fieldTypeIter != intoFields.end());

      // If the number of items isn't specified, just use the type.
      if (field.getNumItems() == 0) {
        fields.push_back({field.getFieldName(), fieldTypeIter->getSecond()});
      } else {
        // If the number of items is specified, we can assume that it's an array
        // type.
        auto array = hw::type_cast<hw::ArrayType>(fieldTypeIter->getSecond());
        assert(fields.empty()); // Checked by the validator.

        // The first union entry should be an array of length numItems.
        fields.push_back(
            {field.getFieldName(),
             hw::ArrayType::get(array.getElementType(), field.getNumItems())});
        unionFields.push_back(
            {frame.getName(), hw::StructType::get(getContext(), fields), 0});
        fields.clear();

        // If the array size is not a multiple of numItems, we need another
        // frame for the left overs.
        size_t leftOver = array.getNumElements() % field.getNumItems();
        if (leftOver) {
          fields.push_back(
              {field.getFieldName(),
               hw::ArrayType::get(array.getElementType(), leftOver)});

          unionFields.push_back(
              {StringAttr::get(getContext(),
                               Twine(frame.getName().getValue(), "_leftOver")),
               hw::StructType::get(getContext(), fields), 0});
          fields.clear();
        }
      }
    }

    if (!fields.empty())
      unionFields.push_back(
          {frame.getName(), hw::StructType::get(getContext(), fields), 0});
  }

  return hw::UnionType::get(getContext(), unionFields);
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
