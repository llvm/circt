//===- EmitUHDI.cpp - UHDI debug info emission ----------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Emits debug information as a UHDI JSON document.
//
//===----------------------------------------------------------------------===//

#include "DebugInfoUtils.h"
#include "circt/Analysis/DebugInfo.h"
#include "circt/Dialect/Comb/CombOps.h"
#include "circt/Dialect/Debug/DebugOps.h"
#include "circt/Dialect/HW/HWOps.h"
#include "circt/Dialect/SV/SVOps.h"
#include "circt/Dialect/Seq/SeqTypes.h"
#include "circt/Support/Version.h"
#include "circt/Target/DebugInfo.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/MapVector.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/JSON.h"
#include <optional>
#include <set>
#include <string>
#include <vector>

namespace comb = circt::comb;
namespace json = llvm::json;
namespace debug = circt::debug;
namespace hw = circt::hw;

static std::optional<llvm::StringRef> getInputPortName(mlir::Value value) {
  auto arg = llvm::dyn_cast<mlir::BlockArgument>(value);
  if (!arg)
    return std::nullopt;

  auto hwModule = llvm::dyn_cast<hw::HWModuleOp>(arg.getOwner()->getParentOp());
  if (!hwModule)
    return std::nullopt;

  return hwModule.getPort(hwModule.getPortIdForInputId(arg.getArgNumber()))
      .getVerilogName();
}

static std::optional<llvm::StringRef> getDeclaredName(mlir::Operation *op) {
  if (!llvm::isa_and_nonnull<hw::WireOp, circt::sv::WireOp, circt::sv::RegOp,
                             circt::sv::LogicOp>(op))
    return std::nullopt;

  if (auto name = op->getAttrOfType<mlir::StringAttr>("hw.verilogName");
      name && !name.empty())
    return name.getValue();
  if (auto name = op->getAttrOfType<mlir::StringAttr>("name");
      name && !name.empty())
    return name.getValue();
  return std::nullopt;
}

static std::optional<llvm::StringRef> getOutputPortName(mlir::Value value) {
  for (auto &use : value.getUses()) {
    if (!llvm::isa<hw::OutputOp>(use.getOwner()))
      continue;
    auto hwModule = llvm::cast<hw::HWModuleOp>(use.getOwner()->getParentOp());
    return hwModule
        .getPort(hwModule.getPortIdForOutputId(use.getOperandNumber()))
        .getVerilogName();
  }
  return std::nullopt;
}

static std::optional<llvm::StringRef> getSignalName(mlir::Value value) {
  // A wire or reg is read through `sv.read_inout`.
  if (auto read = value.getDefiningOp<circt::sv::ReadInOutOp>(); read)
    value = read.getInput();
  if (auto name = getInputPortName(value); name)
    return name;
  if (auto name = getDeclaredName(value.getDefiningOp()); name)
    return name;
  return getOutputPortName(value);
}

/// All bits of `value` as `0b...`, most significant first.
static std::string getBits(const llvm::APInt &value) {
  std::string bits = "0b";
  bits.reserve(2 + value.getBitWidth());
  for (unsigned i = value.getBitWidth(); i-- > 0;)
    bits += value[i] ? '1' : '0';
  return bits;
}

static void emitOptimizedOut(json::OStream &out) {
  out.object([&] { out.attribute("$optimized_out", true); });
}

/// The low bit of the field that `op` selects in its input.
static std::optional<int64_t> getLowBit(hw::StructExtractOp op) {
  // A packed struct keeps its first field in the most significant bits.
  auto fields =
      hw::type_cast<hw::StructType>(op.getInput().getType()).getElements();
  int64_t lowBit = 0;
  for (auto field : fields.drop_front(op.getFieldIndex() + 1)) {
    auto width = hw::getBitWidth(field.type);
    if (width < 0)
      return std::nullopt;

    lowBit += width;
  }

  return lowBit;
}

/// The low bit of the element that `op` selects in its input,
/// if the index is a constant.
static std::optional<int64_t> getLowBit(hw::ArrayGetOp op) {
  auto index = op.getIndex().getDefiningOp<hw::ConstantOp>();
  if (!index)
    return std::nullopt;

  // A packed array keeps element 0 in the least significant bits.
  auto elementType =
      hw::type_cast<hw::ArrayType>(op.getInput().getType()).getElementType();
  auto width = hw::getBitWidth(elementType);
  if (width < 0)
    return std::nullopt;

  return index.getValue().getZExtValue() * width;
}

/// A signal and the low bit of a run of its bits.
using SignalBits = std::pair<llvm::StringRef, int64_t>;

/// The signal that holds `value` as a run of its bits.
/// A field or a constant index of a packed aggregate
/// selects bits of the aggregate's signal.
static std::optional<SignalBits> getSignalBits(mlir::Value value) {
  if (auto signal = getSignalName(value); signal)
    return std::make_pair(*signal, int64_t(0));

  auto *op = value.getDefiningOp();
  if (!op)
    return std::nullopt;

  return llvm::TypeSwitch<mlir::Operation *, std::optional<SignalBits>>(op)
      .Case<hw::StructExtractOp, hw::ArrayGetOp>(
          [](auto op) -> std::optional<SignalBits> {
            auto lowBit = getLowBit(op);
            auto bits = getSignalBits(op.getInput());
            if (!lowBit || !bits)
              return std::nullopt;

            bits->second += *lowBit;
            return bits;
          })
      .Case([](comb::ExtractOp op) -> std::optional<SignalBits> {
        auto bits = getSignalBits(op.getInput());
        if (bits)
          bits->second += op.getLowBit();
        return bits;
      })
      .Default(std::optional<SignalBits>());
}

/// Emit the value of a variable: a signal, a literal,
/// or an aggregate of those shaped like the variable's type.
static void emitValue(json::OStream &out, mlir::Value value) {
  if (!value)
    return emitOptimizedOut(out);

  if (auto signal = getSignalName(value); signal)
    return out.value(*signal);

  auto *op = value.getDefiningOp();
  if (!op)
    return emitOptimizedOut(out);

  llvm::TypeSwitch<mlir::Operation *>(op)
      .Case([&](debug::ValueOp op) { emitValue(out, op.getValue()); })
      .Case([&](debug::EnumOp op) { emitValue(out, op.getValue()); })
      .Case([&](debug::StructOp op) {
        out.object([&] {
          for (auto [name, field] :
               llvm::zip(op.getNames().getAsValueRange<mlir::StringAttr>(),
                         op.getFields())) {
            out.attributeBegin(name);
            emitValue(out, field);
            out.attributeEnd();
          }
        });
      })
      .Case([&](debug::ArrayOp op) {
        out.array([&] {
          for (auto element : op.getElements())
            emitValue(out, element);
        });
      })
      .Case<hw::StructExtractOp, hw::ArrayGetOp, comb::ExtractOp>([&](auto) {
        auto bits = getSignalBits(value);
        auto width = hw::getBitWidth(value.getType());
        if (!bits || width < 0)
          return emitOptimizedOut(out);

        out.object([&] {
          out.attribute("$slice", bits->first);
          out.attribute("hi", bits->second + width - 1);
          out.attribute("lo", bits->second);
        });
      })
      .Case([&](hw::ConstantOp op) {
        out.object([&] {
          // A zero-width value has no bits to give.
          if (op.getValue().getBitWidth() == 0)
            out.attribute("$zero_width", true);
          else
            out.attribute("$const", getBits(op.getValue()));
        });
      })
      // Expressions (`$op`) are not emitted yet:
      // no signal holds the value of any other op.
      .Default([&](mlir::Operation *) { emitOptimizedOut(out); });
}

static void emitPath(json::OStream &out, llvm::StringRef key,
                     llvm::ArrayRef<llvm::StringRef> path) {
  out.attributeArray(key, [&] {
    for (auto name : path)
      out.value(name);
  });
}

/// The variable name of an instance.
/// An artificial instance has no name in the source, so it keeps
/// its Verilog name, which no other variable can have.
static llvm::StringRef getVarName(circt::DIInstance &instance) {
  if (instance.module->isArtificial)
    return circt::getVerilogInstanceName(instance).getValue();
  return instance.name.getValue();
}

/// The key of each variable and instance in the `vars` of its module.
/// A key is unique in its module and has no `.`, `[` or `]`,
/// which the path grammar uses.
using VarKeys = llvm::DenseMap<const void *, std::string>;

static void assignKey(VarKeys &keys, llvm::StringSet<> &used, const void *node,
                      llvm::StringRef name) {
  std::string key = name.str();
  llvm::replace(key, '.', '_');
  llvm::replace(key, '[', '_');
  llvm::replace(key, ']', '_');

  std::string unique = key;
  for (unsigned n = 1; !used.insert(unique).second; ++n)
    unique = (key + "_" + llvm::Twine(n)).str();

  keys[node] = std::move(unique);
}

static void assignKeys(VarKeys &keys, circt::DIModule &diModule) {
  llvm::StringSet<> used;
  for (auto *var : diModule.variables)
    assignKey(keys, used, var, var->name.getValue());
  for (auto *instance : diModule.instances)
    assignKey(keys, used, instance, getVarName(*instance));
}

/// Emit `name` when the key of a variable is not its declared name.
static void emitDeclaredName(json::OStream &out, llvm::StringRef key,
                             llvm::StringRef name) {
  if (key != name)
    out.attribute("name", name);
}

using ModuleMap = llvm::MapVector<mlir::StringAttr, circt::DIModule *>;

/// Every module as written in the source:
/// the modules in the IR and the modules inlined into them.
/// An inlined module appears once, however many times it was inlined.
static ModuleMap collectOriginals(circt::DebugInfo &di) {
  ModuleMap originals;
  for (auto &entry : di.moduleNodes)
    if (!entry.second->isLowered)
      originals.insert(entry);

  // Index, not iterators: `insert` may reallocate,
  // and the loop must also visit the modules it appends.
  for (unsigned i = 0; i < originals.size(); ++i)
    for (auto *instance : originals.getArrayRef()[i].second->instances)
      if (instance->module->isInline)
        originals.insert({instance->module->name, instance->module});

  return originals;
}

/// Call `fn` with the `x-<lang>` key and the JSON of each language
/// in the `ext` of a debug op. A language whose JSON does not parse is left
/// out: the intrinsics reject it, so only hand-written IR has it.
static void forEachExt(mlir::DictionaryAttr ext,
                       llvm::function_ref<void(std::string, json::Value)> fn) {
  if (!ext)
    return;

  for (auto entry : ext) {
    auto text = llvm::cast<mlir::StringAttr>(entry.getValue()).getValue();
    auto value = json::parse(text);
    if (!value) {
      llvm::consumeError(value.takeError());
      continue;
    }
    fn(("x-" + entry.getName().getValue()).str(), std::move(*value));
  }
}

/// Add the `x-<lang>` keys of `ext` to `type`.
static void addExt(json::Object &type, mlir::DictionaryAttr ext) {
  forEachExt(ext, [&](std::string key, json::Value value) {
    type[std::move(key)] = std::move(value);
  });
}

/// The type of a value or an MLIR type, as JSON that nests the types
/// it is built from; `TypeTable::add` interns them.
/// None if it cannot be shown.
static std::optional<json::Object> getType(mlir::Value value);
static std::optional<json::Object> getType(mlir::Type type);

static json::Object getUIntType(int64_t width) {
  return json::Object{{"kind", "uint"}, {"width", width}};
}

static json::Object getStructType(json::Array fields) {
  return json::Object{{"kind", "struct"}, {"fields", std::move(fields)}};
}

static json::Object getVectorType(json::Object elem, size_t size) {
  return json::Object{
      {"kind", "vector"}, {"elem", std::move(elem)}, {"size", size}};
}

static std::optional<json::Object> getStructType(debug::StructOp op) {
  json::Array fields;
  auto flips = op.getFlips().value_or(llvm::ArrayRef<bool>{});
  for (auto [index, name, field] :
       llvm::enumerate(op.getNames(), op.getFields())) {
    auto type = getType(field);
    if (!type)
      return std::nullopt;

    json::Object entry{{"name", llvm::cast<mlir::StringAttr>(name).getValue()},
                       {"type", std::move(*type)}};
    // A flip is part of the type: it changes the entry's key.
    if (index < flips.size() && flips[index])
      entry["flip"] = true;
    fields.push_back(std::move(entry));
  }

  return getStructType(std::move(fields));
}

static std::optional<json::Object> getArrayType(debug::ArrayOp op) {
  auto elements = op.getElements();
  if (elements.empty())
    return std::nullopt;

  auto elem = getType(elements.front());
  if (!elem)
    return std::nullopt;

  return getVectorType(std::move(*elem), elements.size());
}

static std::optional<json::Object> getEnumType(debug::EnumOp op) {
  auto base = getType(op.getValue().getType());
  if (!base)
    return std::nullopt;

  json::Object values;
  for (auto variant : op.getVariantsMap()) {
    auto tag = llvm::cast<mlir::IntegerAttr>(variant.getValue()).getValue();
    // A JSON number holds at most 64 bits; a wider tag is written as a string.
    values[variant.getName().getValue()] =
        tag.getActiveBits() <= 64 ? json::Value(tag.getZExtValue())
                                  : json::Value(llvm::toString(tag, 10, false));
  }

  json::Object type{{"kind", "enum"},
                    {"base", std::move(*base)},
                    {"values", std::move(values)},
                    {"name", op.getEnumTypeName()}};
  addExt(type, op.getExtAttr());
  return type;
}

/// The type of a value with a `dbg.value` kind; a kind may add parameters.
static json::Object getKindType(debug::ValueOp op, debug::ValueKind kind) {
  json::Object type{{"kind", debug::stringifyValueKind(kind)}};
  if (kind == debug::ValueKind::SInt)
    type["width"] = hw::getBitWidth(op.getValue().getType());
  return type;
}

static std::optional<json::Object> getType(mlir::Value value) {
  if (!value)
    return std::nullopt;

  auto *op = value.getDefiningOp();
  if (!op)
    return getType(value.getType());

  return llvm::TypeSwitch<mlir::Operation *, std::optional<json::Object>>(op)
      .Case([&](debug::ValueOp op) -> std::optional<json::Object> {
        auto kind = op.getKind();
        auto type = kind ? getKindType(op, *kind) : getType(op.getValue());
        if (!type)
          return std::nullopt;

        // The source name and language data, merged before the type
        // is interned, so no unnamed copy is left.
        if (auto name = op.getTypeNameAttr())
          (*type)["name"] = name.getValue();
        addExt(*type, op.getExtAttr());

        return type;
      })
      .Case([&](debug::StructOp op) { return getStructType(op); })
      .Case([&](debug::ArrayOp op) { return getArrayType(op); })
      .Case([&](debug::EnumOp op) { return getEnumType(op); })
      .Default([&](mlir::Operation *) { return getType(value.getType()); });
}

static std::optional<json::Object> getType(mlir::Type type) {
  return llvm::TypeSwitch<mlir::Type, std::optional<json::Object>>(
             hw::getCanonicalType(type))
      .Case(
          [&](hw::InOutType inout) { return getType(inout.getElementType()); })
      .Case([&](mlir::IntegerType intType) {
        return getUIntType(intType.getWidth());
      })
      .Case([&](circt::seq::ClockType) {
        return json::Object{{"kind", "clock"}};
      })
      .Case([&](hw::StructType structType) -> std::optional<json::Object> {
        json::Array fields;
        for (auto field : structType.getElements()) {
          auto type = getType(field.type);
          if (!type)
            return std::nullopt;

          fields.push_back(json::Object{{"name", field.name.getValue()},
                                        {"type", std::move(*type)}});
        }
        return getStructType(std::move(fields));
      })
      .Case<hw::ArrayType, hw::UnpackedArrayType>(
          [&](auto arrayType) -> std::optional<json::Object> {
            auto elem = getType(arrayType.getElementType());
            if (!elem)
              return std::nullopt;

            return getVectorType(std::move(*elem), arrayType.getNumElements());
          })
      .Default([&](mlir::Type type) -> std::optional<json::Object> {
        // Show any other type as its bits.
        if (auto width = hw::getBitWidth(type); width >= 0)
          return getUIntType(width);
        return std::nullopt;
      });
}

namespace {
/// The `types` table. Equal types share one entry,
/// and a type comes after the types it is built from.
struct TypeTable final {
  void addModule(circt::DIModule &diModule);
  void addExternPorts(circt::DIModule &diModule);
  unsigned add(json::Object type);

  std::vector<json::Value> types;
  llvm::StringMap<unsigned> indices;
  llvm::DenseMap<circt::DIVariable *, unsigned> varTypes;
  llvm::DenseMap<circt::DIInstance *, unsigned> instanceTypes;
};
} // namespace

void TypeTable::addModule(circt::DIModule &diModule) {
  if (diModule.isExtern)
    return addExternPorts(diModule);

  for (auto *var : diModule.variables)
    if (auto type = getType(var->value); type)
      varTypes[var] = add(std::move(*type));

  // After dedup the instances of a module may come from different classes,
  // so each class gets its own type.
  for (auto *instance : diModule.instances) {
    json::Object type{{"kind", "module"},
                      {"module", instance->module->name.getValue()}};
    if (instance->typeName)
      type["name"] = instance->typeName.getValue();
    addExt(type, instance->ext);

    instanceTypes[instance] = add(std::move(type));
  }
}

/// The variables of an external module are its ports, in order.
void TypeTable::addExternPorts(circt::DIModule &diModule) {
  auto ext = llvm::cast<hw::HWModuleExternOp>(diModule.op);
  for (auto [var, port] : llvm::zip(diModule.variables, ext.getPortList()))
    if (auto type = getType(port.type); type)
      varTypes[var] = add(std::move(*type));
}

/// Add `type` after the types nested in it, which become indices,
/// and return its index.
unsigned TypeTable::add(json::Object type) {
  for (auto key : {"base", "elem"})
    if (auto *nested = type.get(key))
      *nested = add(std::move(*nested->getAsObject()));

  if (auto *fields = type.getArray("fields"))
    for (auto &field : *fields) {
      auto *nested = field.getAsObject()->get("type");
      *nested = add(std::move(*nested->getAsObject()));
    }

  json::Value value(std::move(type));
  std::string key;
  llvm::raw_string_ostream(key) << value;

  auto [it, inserted] = indices.try_emplace(key, types.size());
  if (inserted)
    types.push_back(std::move(value));
  return it->second;
}

namespace {
struct FileTable final {
  void emitLoc(json::OStream &out, mlir::Location loc, bool emitted);
  void emitLoc(json::OStream &out, mlir::FileLineColLoc fileLoc, bool emitted);

  /// Path to whether it is a file of the emitted netlist,
  /// in the order the emitter first meets it;
  /// the position is the file number in `loc`.
  llvm::MapVector<llvm::StringRef, bool> paths;
};
} // namespace

/// Emit `[file, line, column]` of the best source or emitted location in `loc`.
void FileTable::emitLoc(json::OStream &out, mlir::Location loc, bool emitted) {
  auto fileLoc = circt::findBestLocation(loc, emitted, /*fileMustExist=*/false);
  if (fileLoc)
    emitLoc(out, fileLoc, emitted);
}

void FileTable::emitLoc(json::OStream &out, mlir::FileLineColLoc fileLoc,
                        bool emitted) {
  auto it = paths.insert({fileLoc.getFilename().getValue(), emitted}).first;
  out.attributeArray("loc", [&] {
    out.value(it - paths.begin());
    out.value(fileLoc.getLine());
    if (auto column = fileLoc.getColumn(); column)
      out.value(column);
  });
}

static void emitTypes(json::OStream &out, TypeTable &table) {
  out.attributeArray("types", [&] {
    for (auto &type : table.types)
      out.value(type);
  });
}

static void emitFiles(json::OStream &out, FileTable &files) {
  out.attributeArray("files", [&] {
    for (auto &file : files.paths)
      out.object([&] {
        out.attribute("path", file.first);
        if (file.second)
          out.attribute("generated", true);
      });
  });
}

/// The location of a module in the source.
/// Dedup fuses the locations of the modules it merges
/// and records each of them in `dbg.module_locs`;
/// the inliner records there the location of an inlined module.
static mlir::LocationAttr getModuleLoc(circt::DIModule &diModule) {
  auto origins = diModule.op->getAttrOfType<mlir::DictionaryAttr>(
      circt::debug::kDbgModuleLocsAttr);
  if (origins)
    if (auto loc = origins.getAs<mlir::LocationAttr>(diModule.name))
      return loc;

  // The op of an inlined module is its `dbg.scope`, which is located
  // at the instance, not at the module; the instance variable gets it.
  if (diModule.isInline)
    return {};

  return diModule.op->getLoc();
}

/// Every `x-` key of the document names a profile it uses.
static void emitProfiles(json::OStream &out, TypeTable &table,
                         ModuleMap &originals) {
  // std::set: the profiles are emitted sorted.
  std::set<std::string> profiles;
  for (auto &type : table.types)
    for (auto &entry : *type.getAsObject())
      if (llvm::StringRef(entry.first).starts_with("x-"))
        profiles.insert(entry.first.str());

  for (auto &entry : originals)
    forEachExt(entry.second->ext, [&](std::string key, json::Value) {
      profiles.insert(std::move(key));
    });

  out.attributeArray("profiles", [&] {
    for (auto &profile : profiles)
      out.value(profile);
  });
}

static void emitOriginalVar(json::OStream &out, circt::DIVariable &var,
                            unsigned type, FileTable &files, VarKeys &keys) {
  emitDeclaredName(out, keys[&var], var.name.getValue());
  out.attribute("type", type);
  files.emitLoc(out, var.loc, /*emitted=*/false);

  // UHDI has no key for a wire: a variable without `storage` or `node` is one.
  if (var.kind == debug::VariableKind::Node)
    out.attribute("node", true);
  else if (var.kind && var.kind != debug::VariableKind::Wire)
    out.attribute("storage", debug::stringifyVariableKind(*var.kind));
  if (var.dir)
    out.attribute("dir", debug::stringifyPortDirection(*var.dir));
}

static void emitOriginalInstance(json::OStream &out,
                                 circt::DIInstance &instance, TypeTable &table,
                                 FileTable &files, VarKeys &keys) {
  emitDeclaredName(out, keys[&instance], getVarName(instance));
  out.attribute("type", table.instanceTypes.lookup(&instance));
  files.emitLoc(out, instance.op->getLoc(), /*emitted=*/false);

  if (instance.module->isArtificial)
    out.attribute("artificial", true);
}

static void emitOriginalModule(json::OStream &out, circt::DIModule &diModule,
                               TypeTable &table, FileTable &files,
                               VarKeys &keys) {
  if (diModule.op)
    if (auto loc = getModuleLoc(diModule))
      files.emitLoc(out, loc, /*emitted=*/false);

  if (diModule.isArtificial)
    out.attribute("artificial", true);

  // The top module has no instance whose type names its class.
  if (auto name = diModule.typeName)
    out.attribute("name", name.getValue());
  forEachExt(diModule.ext, [&](std::string key, json::Value value) {
    out.attribute(key, value);
  });

  out.attributeObject("vars", [&] {
    for (auto *var : diModule.variables) {
      // A variable of a type that cannot be shown is left out.
      auto it = table.varTypes.find(var);
      if (it == table.varTypes.end())
        continue;

      out.attributeObject(keys[var], [&] {
        emitOriginalVar(out, *var, it->second, files, keys);
      });
    }

    for (auto *instance : diModule.instances)
      out.attributeObject(keys[instance], [&] {
        emitOriginalInstance(out, *instance, table, files, keys);
      });
  });
}

static void emitOriginal(json::OStream &out, ModuleMap &originals,
                         TypeTable &table, FileTable &files, VarKeys &keys) {
  out.attributeObject("original", [&] {
    for (auto &entry : originals)
      out.attributeObject(entry.first.getValue(), [&] {
        emitOriginalModule(out, *entry.second, table, files, keys);
      });
  });
}

/// Emit where each signal of `hwModule` is declared in the netlist.
/// Of signals with the same name, only the first is kept.
static void emitSignals(json::OStream &out, hw::HWModuleOp hwModule,
                        FileTable &files) {
  llvm::SmallVector<std::pair<llvm::StringRef, mlir::FileLineColLoc>> signals;
  llvm::StringSet<> seen;
  hwModule.walk([&](mlir::Operation *op) {
    auto name = getDeclaredName(op);
    if (!name)
      return;

    auto fileLoc = circt::findBestLocation(op->getLoc(), /*emitted=*/true,
                                           /*fileMustExist=*/false);
    if (fileLoc && seen.insert(*name).second)
      signals.push_back({*name, fileLoc});
  });
  if (signals.empty())
    return;

  out.attributeObject("signals", [&] {
    for (auto &signal : signals)
      out.attributeObject(signal.first, [&] {
        files.emitLoc(out, signal.second, /*emitted=*/true);
      });
  });
}

static void emitGenerated(json::OStream &out, circt::DebugInfo &di,
                          FileTable &files) {
  // External modules may share one Verilog module.
  llvm::SmallDenseSet<mlir::StringAttr> seen;
  out.attributeObject("generated", [&] {
    for (auto &entry : di.moduleNodes) {
      circt::DIModule &diModule = *entry.second;
      auto name = circt::getVerilogModuleName(diModule);
      if (!seen.insert(name).second)
        continue;

      out.attributeObject(name.getValue(), [&] {
        if (auto *op = diModule.op; op)
          files.emitLoc(out, op->getLoc(), /*emitted=*/true);
        if (diModule.isExtern)
          out.attribute("kind", "extern");

        auto hwModule = llvm::dyn_cast_or_null<hw::HWModuleOp>(diModule.op);
        if (hwModule)
          emitSignals(out, hwModule, files);
      });
    }
  });
}

/// An external module has no body; its ports keep their names.
static void emitExternPorts(json::OStream &out, circt::DIModule &diModule,
                            TypeTable &table, VarKeys &keys) {
  auto ext = llvm::cast<hw::HWModuleExternOp>(diModule.op);
  for (auto [var, port] : llvm::zip(diModule.variables, ext.getPortList()))
    if (table.varTypes.contains(var))
      out.attribute(keys[var], port.getVerilogName());
}

static void emitInstanceRef(json::OStream &out, circt::DIInstance &instance,
                            llvm::SmallVectorImpl<llvm::StringRef> &inlinePath,
                            VarKeys &keys) {
  out.attributeObject(keys[&instance], [&] {
    if (instance.module->isInline) {
      inlinePath.push_back(instance.name.getValue());
      emitPath(out, "$inline", inlinePath);
      inlinePath.pop_back();
      return;
    }

    out.attribute("$inst", circt::getVerilogInstanceName(instance).getValue());
    out.attribute("module",
                  circt::getVerilogModuleName(*instance.module).getValue());
  });
}

/// Emit how the variables of `diModule` map to the signals of the Verilog
/// module `generated`, followed by the bindings of the modules inlined into it.
/// Modules merged by dedup that inline the same module would repeat
/// its binding; `seen` keeps only the first.
static void emitBinding(json::OStream &out, circt::DIModule &diModule,
                        llvm::StringRef generated,
                        llvm::SmallVectorImpl<llvm::StringRef> &inlinePath,
                        llvm::StringSet<> &seen, TypeTable &table,
                        VarKeys &keys) {
  std::string key = (diModule.name.getValue() + "/" + generated).str();
  for (auto name : inlinePath)
    key += ("/" + name).str();

  if (!seen.insert(key).second)
    return;

  out.object([&] {
    out.attribute("original", diModule.name.getValue());
    out.attribute("generated", generated);
    if (!inlinePath.empty())
      emitPath(out, "inline_path", inlinePath);

    out.attributeObject("vars", [&] {
      if (diModule.isExtern) {
        emitExternPorts(out, diModule, table, keys);
        return;
      }

      for (auto *var : diModule.variables) {
        if (!table.varTypes.contains(var))
          continue;

        out.attributeBegin(keys[var]);
        emitValue(out, var->value);
        out.attributeEnd();
      }

      for (auto *instance : diModule.instances)
        emitInstanceRef(out, *instance, inlinePath, keys);
    });
  });

  for (auto *instance : diModule.instances)
    if (instance->module->isInline) {
      inlinePath.push_back(instance->name.getValue());
      emitBinding(out, *instance->module, generated, inlinePath, seen, table,
                  keys);
      inlinePath.pop_back();
    }
}

static void emitBindings(json::OStream &out, circt::DebugInfo &di,
                         TypeTable &table, VarKeys &keys) {
  llvm::SmallVector<llvm::StringRef> inlinePath;
  llvm::StringSet<> seen;
  out.attributeArray("bindings", [&] {
    for (auto &entry : di.moduleNodes) {
      if (entry.second->isLowered)
        continue;

      auto generated = circt::getVerilogModuleName(*entry.second).getValue();
      emitBinding(out, *entry.second, generated, inlinePath, seen, table, keys);
    }
  });
}

/// A module made by lowering is not in the source, nor are its instances.
/// It stays in `generated`.
static void dropLowered(circt::DIModule &module) {
  llvm::erase_if(module.instances,
                 [](auto *instance) { return instance->module->isLowered; });
  for (auto *instance : module.instances)
    if (instance->module->isInline)
      dropLowered(*instance->module);
}

/// Give each variable and instance of `copy` the key of the one
/// with the same name in `original`. A name `original` lacks gets a new key.
static void copyKeys(VarKeys &keys, circt::DIModule &copy,
                     circt::DIModule &original) {
  // Same-named entries take the keys in order.
  llvm::StringMap<llvm::SmallVector<std::string, 1>> varKeys, instanceKeys;
  llvm::StringSet<> used;
  for (auto *var : original.variables) {
    varKeys[var->name.getValue()].push_back(keys[var]);
    used.insert(keys[var]);
  }
  for (auto *instance : original.instances) {
    instanceKeys[getVarName(*instance)].push_back(keys[instance]);
    used.insert(keys[instance]);
  }

  auto take = [&](auto &byName, const void *node, llvm::StringRef name) {
    auto &free = byName[name];
    if (free.empty())
      return assignKey(keys, used, node, name);

    keys[node] = free.front();
    free.erase(free.begin());
  };

  for (auto *var : copy.variables)
    take(varKeys, var, var->name.getValue());
  for (auto *instance : copy.instances)
    take(instanceKeys, instance, getVarName(*instance));
}

/// Each inlined copy of a module has its own variables, which its binding
/// needs. A copy gets the keys of the module in `original`.
static void addInlinedCopies(circt::DIModule &module, ModuleMap &originals,
                             TypeTable &table, VarKeys &keys) {
  for (auto *instance : module.instances) {
    auto *copy = instance->module;
    if (!copy->isInline)
      continue;

    table.addModule(*copy);
    if (auto *original = originals.lookup(copy->name))
      copyKeys(keys, *copy, *original);
    else
      assignKeys(keys, *copy);
    addInlinedCopies(*copy, originals, table, keys);
  }
}

mlir::LogicalResult circt::debug::emitUHDI(mlir::Operation *op,
                                           llvm::raw_ostream &os,
                                           unsigned indent) {
  circt::DebugInfo di(op);
  for (auto &entry : di.moduleNodes)
    dropLowered(*entry.second);

  auto originals = collectOriginals(di);
  TypeTable table;
  VarKeys keys;
  for (auto &entry : originals) {
    table.addModule(*entry.second);
    assignKeys(keys, *entry.second);
  }

  for (auto &entry : di.moduleNodes)
    addInlinedCopies(*entry.second, originals, table, keys);

  FileTable files;

  json::OStream out(os, indent);
  out.object([&] {
    out.attribute("uhdi", "0.1");
    out.attribute("producer", circt::getCirctVersion());
    emitProfiles(out, table, originals);
    emitTypes(out, table);
    emitOriginal(out, originals, table, files, keys);
    emitGenerated(out, di, files);
    emitBindings(out, di, table, keys);
    // Files come last: the sections above number them as they meet them.
    emitFiles(out, files);
  });
  os << "\n";
  return mlir::success();
}
