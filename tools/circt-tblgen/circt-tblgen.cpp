//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "circt/Support/Version.h"
#include "mlir/TableGen/GenInfo.h"
#include "mlir/Tools/mlir-tblgen/MlirTblgenMain.h"
#include "llvm/Support/PrettyStackTrace.h"
#include "llvm/TableGen/Record.h"

using namespace llvm;
using namespace mlir;
using namespace circt;

// Generator that prints records.
static GenRegistration
    printRecords("print-records", "Print all records to stdout",
                 [](const RecordKeeper &records, raw_ostream &os) {
                   os << records;
                   return false;
                 });

int main(int argc, char **argv) {
  // Set the bug report message to indicate users should file issues on
  // llvm/circt and not llvm/llvm-project.
  setBugReportMsg(circtBugReportMsg);

  return MlirTblgenMain(argc, argv);
}
