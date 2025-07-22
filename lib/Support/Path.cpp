//===- Path.cpp - Path Utilities --------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Utilities for file system path handling, supplementing the ones from
// llvm::sys::path.
//
//===----------------------------------------------------------------------===//

#include "circt/Support/Path.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/Support/FileUtilities.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Path.h"

using namespace circt;

/// Append a path to an existing path, replacing it if the other path is
/// absolute. This mimicks the behaviour of `foo/bar` and `/foo/bar` being used
/// in a working directory `/home`, resulting in `/home/foo/bar` and `/foo/bar`,
/// respectively.
void circt::appendPossiblyAbsolutePath(llvm::SmallVectorImpl<char> &base,
                                       const llvm::Twine &suffix) {
  if (llvm::sys::path::is_absolute(suffix)) {
    base.clear();
    suffix.toVector(base);
  } else {
    llvm::sys::path::append(base, suffix);
  }
}

std::unique_ptr<llvm::ToolOutputFile>
circt::createOutputFile(StringRef filename, StringRef dirname,
                        function_ref<InFlightDiagnostic()> emitError) {
  // Determine the output path from the output directory and filename.
  SmallString<128> outputFilename(dirname);
  appendPossiblyAbsolutePath(outputFilename, filename);
  auto outputDir = llvm::sys::path::parent_path(outputFilename);

  // Create the output directory if needed.
  std::error_code error = llvm::sys::fs::create_directories(outputDir);
  if (error) {
    emitError() << "cannot create output directory \"" << outputDir
                << "\": " << error.message();
    return {};
  }

  // Open the output file.
  std::string errorMessage;
  auto output = mlir::openOutputFile(outputFilename, &errorMessage);
  if (!output)
    emitError() << errorMessage;
  return output;
}
