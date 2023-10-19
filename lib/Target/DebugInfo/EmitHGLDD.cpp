//===- EmitHGLDD.cpp - HGLDD debug info emission --------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "circt/Analysis/DebugInfo.h"
#include "circt/Dialect/HW/HWOps.h"
#include "circt/Target/DebugInfo.h"
#include "mlir/IR/Threading.h"
#include "mlir/Support/FileUtilities.h"
#include "mlir/Support/IndentedOstream.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/ToolOutputFile.h"

#define DEBUG_TYPE "di"

using namespace mlir;
using namespace circt;
using namespace debug;

using llvm::MapVector;
using llvm::SmallMapVector;

/// Walk the given `loc` and collect file-line-column locations that we want to
/// report as source ("HGL") locations or as emitted Verilog ("HDL") locations.
///
/// This function treats locations inside a `NameLoc` called "emitted" or a
/// `FusedLoc` with the metadata attribute string "verilogLocations" as emitted
/// Verilog locations. All other locations are considered to be source
/// locations.
///
/// The `level` parameter is used to track into how many "emitted" or
/// "verilogLocations" we have already descended. For every one of those we look
/// through the level gets decreased by one. File-line-column locations are only
/// collected at level 0. We don't descend into "emitted" or "verilogLocations"
/// once we've reached level 0. This effectively makes the `level` parameter
/// decide behind how many layers of "emitted" or "verilogLocations" we want to
/// collect file-line-column locations. Setting this to 0 effectively collects
/// source locations, i.e., everything not marked as emitted. Setting this to 1
/// effectively collects emitted locations, i.e., nothing that isn't behind
/// exactly one layer of "emitted" or "verilogLocations".
static void findLocations(Location loc, unsigned level,
                          SmallVectorImpl<FileLineColLoc> &locs) {
  if (auto nameLoc = dyn_cast<NameLoc>(loc)) {
    if (nameLoc.getName() == "emitted")
      if (level-- == 0)
        return;
    findLocations(nameLoc.getChildLoc(), level, locs);
  } else if (auto fusedLoc = dyn_cast<FusedLoc>(loc)) {
    auto strAttr = dyn_cast_or_null<StringAttr>(fusedLoc.getMetadata());
    if (strAttr && strAttr.getValue() == "verilogLocations")
      if (level-- == 0)
        return;
    for (auto innerLoc : fusedLoc.getLocations())
      findLocations(innerLoc, level, locs);
  } else if (auto fileLoc = dyn_cast<FileLineColLoc>(loc)) {
    if (level == 0)
      locs.push_back(fileLoc);
  }
}

/// Find the best location to report as source location ("HGL", emitted = false)
/// or as emitted location ("HDL", emitted = true). Returns any non-FIR file it
/// finds, and only falls back to FIR files if nothing else is found.
static FileLineColLoc findBestLocation(Location loc, bool emitted) {
  SmallVector<FileLineColLoc> locs;
  findLocations(loc, emitted ? 1 : 0, locs);
  for (auto loc : locs)
    if (!loc.getFilename().getValue().endswith(".fir"))
      return loc;
  for (auto loc : locs)
    if (loc.getFilename().getValue().endswith(".fir"))
      return loc;
  return {};
}

//===----------------------------------------------------------------------===//
// HGLDD File Emission
//===----------------------------------------------------------------------===//

namespace {

/// Contextual information for a single HGLDD file to be emitted.
struct FileEmitter {
  const EmitHGLDDOptions *options = nullptr;
  SmallVector<DIModule *> modules;
  SmallString<64> outputFileName;
  StringAttr hdlFile;
  SmallMapVector<StringAttr, unsigned, 8> sourceFiles;

  void emit(llvm::raw_ostream &os);
  void emit(llvm::json::OStream &json);
  void emitLoc(llvm::json::OStream &json, FileLineColLoc loc, bool emitted);
  void emitModule(llvm::json::OStream &json, DIModule *module);
  void emitInstance(llvm::json::OStream &json, DIInstance *instance);
  void emitVariable(llvm::json::OStream &json, DIVariable *variable);

  /// Get a numeric index for the given `sourceFile`. Populates `sourceFiles`
  /// with a unique ID assignment for each source file.
  unsigned getSourceFile(StringAttr sourceFile, bool emitted) {
    // Apply the source file prefix if this is a source file (emitted = false).
    if (!emitted && !options->sourceFilePrefix.empty() &&
        !llvm::sys::path::is_absolute(sourceFile.getValue())) {
      SmallString<64> buffer;
      buffer = options->sourceFilePrefix;
      llvm::sys::path::append(buffer, sourceFile.getValue());
      sourceFile = StringAttr::get(sourceFile.getContext(), buffer);
    }

    // Apply the output file prefix if this is an outpu file (emitted = true).
    if (emitted && !options->outputFilePrefix.empty() &&
        !llvm::sys::path::is_absolute(sourceFile.getValue())) {
      SmallString<64> buffer;
      buffer = options->outputFilePrefix;
      llvm::sys::path::append(buffer, sourceFile.getValue());
      sourceFile = StringAttr::get(sourceFile.getContext(), buffer);
    }

    auto &slot = sourceFiles[sourceFile];
    if (slot == 0)
      slot = sourceFiles.size();
    return slot;
  }

  /// Find the best location and, if one is found, emit it under the given
  /// `fieldName`.
  void findAndEmitLoc(llvm::json::OStream &json, StringRef fieldName,
                      Location loc, bool emitted) {
    if (auto fileLoc = findBestLocation(loc, emitted))
      json.attributeObject(fieldName, [&] { emitLoc(json, fileLoc, emitted); });
  }
};

} // namespace

void FileEmitter::emit(llvm::raw_ostream &os) {
  llvm::json::OStream json(os, 2);
  emit(json);
  os << "\n";
}

void FileEmitter::emit(llvm::json::OStream &json) {
  // The "HGLDD" header field needs to be the first in the JSON file (which
  // violates the JSON spec, but what can you do). But we only know after module
  // emission what the contents of the header will be.
  std::string rawObjects;
  {
    llvm::raw_string_ostream objectsOS(rawObjects);
    llvm::json::OStream objectsJson(objectsOS, 2);
    objectsJson.arrayBegin(); // dummy for indentation
    objectsJson.arrayBegin();
    for (auto *module : modules)
      emitModule(objectsJson, module);
    objectsJson.arrayEnd();
    objectsJson.arrayEnd(); // dummy for indentation
  }

  std::optional<unsigned> hdlFileIndex;
  if (hdlFile)
    hdlFileIndex = getSourceFile(hdlFile, true);

  json.objectBegin();

  json.attributeObject("HGLDD", [&] {
    json.attribute("version", "1.0");
    json.attributeArray("file_info", [&] {
      for (auto [file, index] : sourceFiles)
        json.value(file.getValue());
    });
    if (hdlFileIndex)
      json.attribute("hdl_file_index", *hdlFileIndex);
  });

  json.attributeBegin("objects");
  json.rawValue(StringRef(rawObjects).drop_front().drop_back().trim());
  json.attributeEnd();

  json.objectEnd();
}

void FileEmitter::emitLoc(llvm::json::OStream &json, FileLineColLoc loc,
                          bool emitted) {
  json.attribute("file", getSourceFile(loc.getFilename(), emitted));
  if (auto line = loc.getLine()) {
    json.attribute("begin_line", line);
    json.attribute("end_line", line);
  }
  if (auto col = loc.getColumn()) {
    json.attribute("begin_column", col);
    json.attribute("end_column", col);
  }
}

StringAttr getVerilogModuleName(DIModule &module) {
  if (auto *op = module.op)
    if (auto attr = op->getAttrOfType<StringAttr>("verilogName"))
      return attr;
  return module.name;
}

StringAttr getVerilogInstanceName(DIInstance &inst) {
  if (auto *op = inst.op)
    if (auto attr = op->getAttrOfType<StringAttr>("hw.verilogName"))
      return attr;
  return inst.name;
}

/// Emit the debug info for a `DIModule`.
void FileEmitter::emitModule(llvm::json::OStream &json, DIModule *module) {
  json.objectBegin();
  json.attribute("kind", "module");
  json.attribute("obj_name", module->name.getValue()); // HGL
  json.attribute("module_name",
                 getVerilogModuleName(*module).getValue()); // HDL
  if (module->isExtern)
    json.attribute("isExtModule", 1);
  if (auto *op = module->op) {
    findAndEmitLoc(json, "hgl_loc", op->getLoc(), false);
    findAndEmitLoc(json, "hdl_loc", op->getLoc(), true);
  }
  json.attributeArray("port_vars", [&] {
    for (auto *var : module->variables)
      emitVariable(json, var);
  });
  json.attributeArray("children", [&] {
    for (auto *instance : module->instances)
      emitInstance(json, instance);
  });
  json.objectEnd();
}

/// Emit the debug info for a `DIInstance`.
void FileEmitter::emitInstance(llvm::json::OStream &json,
                               DIInstance *instance) {
  json.objectBegin();
  json.attribute("name", instance->name.getValue());
  auto verilogName = getVerilogInstanceName(*instance);
  if (verilogName != instance->name)
    json.attribute("hdl_obj_name", verilogName.getValue());
  json.attribute("obj_name", instance->module->name.getValue()); // HGL
  json.attribute("module_name",
                 getVerilogModuleName(*instance->module).getValue()); // HDL
  if (auto *op = instance->op) {
    findAndEmitLoc(json, "hgl_loc", op->getLoc(), false);
    findAndEmitLoc(json, "hdl_loc", op->getLoc(), true);
  }
  json.objectEnd();
}

/// Emit the debug info for a `DIVariable`.
void FileEmitter::emitVariable(llvm::json::OStream &json,
                               DIVariable *variable) {
  json.objectBegin();
  json.attribute("var_name", variable->name.getValue());
  findAndEmitLoc(json, "hgl_loc", variable->loc, false);
  findAndEmitLoc(json, "hdl_loc", variable->loc, true);

  if (auto value = variable->value) {
    StringAttr portName;
    auto *defOp = value.getParentBlock()->getParentOp();
    auto module = dyn_cast<hw::HWModuleOp>(defOp);
    if (!module)
      module = defOp->getParentOfType<hw::HWModuleOp>();
    if (module) {
      if (auto arg = dyn_cast<BlockArgument>(value)) {
        portName = dyn_cast_or_null<StringAttr>(
            module.getInputNames()[arg.getArgNumber()]);
      } else if (auto wireOp = value.getDefiningOp<hw::WireOp>()) {
        portName = wireOp.getNameAttr();
      } else {
        for (auto &use : value.getUses()) {
          if (auto outputOp = dyn_cast<hw::OutputOp>(use.getOwner())) {
            portName = dyn_cast_or_null<StringAttr>(
                module.getOutputNames()[use.getOperandNumber()]);
            break;
          }
        }
      }
    }
    if (auto intType = dyn_cast<IntegerType>(value.getType())) {
      json.attribute("type_name", "logic");
      if (intType.getIntOrFloatBitWidth() != 1) {
        json.attributeArray("packed_range", [&] {
          json.value(intType.getIntOrFloatBitWidth() - 1);
          json.value(0);
        });
      }
    }
    if (portName) {
      json.attributeObject(
          "value", [&] { json.attribute("sig_name", portName.getValue()); });
    }
  }

  json.objectEnd();
}

//===----------------------------------------------------------------------===//
// Output Splitting
//===----------------------------------------------------------------------===//

namespace {

/// Contextual information for HGLDD emission shared across multiple HGLDD
/// files. This struct is used to determine an initial split of debug info files
/// and to distribute work.
struct Emitter {
  DebugInfo di;
  SmallVector<FileEmitter, 0> files;

  Emitter(Operation *module, const EmitHGLDDOptions &options);
};

} // namespace

Emitter::Emitter(Operation *module, const EmitHGLDDOptions &options)
    : di(module) {
  // Group the DI modules according to their emitted file path. Modules that
  // don't have an emitted file path annotated are collected in a separate
  // group. That group, with a null `StringAttr` key, is emitted into a separate
  // "global.dd" file.
  MapVector<StringAttr, FileEmitter> groups;
  for (auto [moduleName, module] : di.moduleNodes) {
    StringAttr hdlFile;
    if (module->op)
      if (auto fileLoc = findBestLocation(module->op->getLoc(), true))
        hdlFile = fileLoc.getFilename();
    groups[hdlFile].modules.push_back(module);
  }

  // Determine the output file names and move the emitters into the `files`
  // member.
  files.reserve(groups.size());
  for (auto &[hdlFile, emitter] : groups) {
    emitter.options = &options;
    emitter.hdlFile = hdlFile;
    emitter.outputFileName = options.outputDirectory;
    StringRef fileName = hdlFile ? hdlFile.getValue() : "global";
    if (llvm::sys::path::is_absolute(fileName))
      emitter.outputFileName = fileName;
    else
      llvm::sys::path::append(emitter.outputFileName, fileName);
    llvm::sys::path::replace_extension(emitter.outputFileName, "dd");
    files.push_back(std::move(emitter));
  }

  // Dump some information about the files to be created.
  LLVM_DEBUG({
    llvm::dbgs() << "HGLDD files:\n";
    for (auto &emitter : files) {
      llvm::dbgs() << "- " << emitter.outputFileName << " (from "
                   << emitter.hdlFile << ")\n";
      for (auto *module : emitter.modules)
        llvm::dbgs() << "  - " << module->name << "\n";
    }
  });
}

//===----------------------------------------------------------------------===//
// Emission Entry Points
//===----------------------------------------------------------------------===//

LogicalResult debug::emitHGLDD(Operation *module, llvm::raw_ostream &os,
                               const EmitHGLDDOptions &options) {
  Emitter emitter(module, options);
  for (auto &fileEmitter : emitter.files) {
    os << "\n// ----- 8< ----- FILE \"" + fileEmitter.outputFileName +
              "\" ----- 8< -----\n\n";
    fileEmitter.emit(os);
  }
  return success();
}

LogicalResult debug::emitSplitHGLDD(Operation *module,
                                    const EmitHGLDDOptions &options) {
  Emitter emitter(module, options);

  auto emit = [&](auto &fileEmitter) {
    // Open the output file for writing.
    std::string errorMessage;
    auto output =
        mlir::openOutputFile(fileEmitter.outputFileName, &errorMessage);
    if (!output) {
      module->emitError(errorMessage);
      return failure();
    }

    // Emit the debug information and keep the file around.
    fileEmitter.emit(output->os());
    output->keep();
    return success();
  };

  return mlir::failableParallelForEach(module->getContext(), emitter.files,
                                       emit);
}
