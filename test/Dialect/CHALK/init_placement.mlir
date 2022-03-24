// RUN: circt-opt --pass-pipeline='chalk.cell(chalk-init-placement)' %s | FileCheck %s --dump-input=fail

// CHECK:  chalk.cell "CombCell" {
// CHECK:  chalk.rectangle "CombRect1" {height = 2 : ui32, width = 1 : ui32, xCoord = 0 : i64, yCoord = 0 : i64}
// CHECK:  chalk.rectangle "CombRect2" {height = 2 : ui32, width = 2 : ui32, xCoord = 1 : i64, yCoord = 0 : i64}
// CHECK:  chalk.rectangle "CombRect3" {height = 2 : ui32, width = 3 : ui32, xCoord = 3 : i64, yCoord = 0 : i64}
// CHECK:  }
// CHECK: }
chalk.cell "CombCell" {} {
    chalk.rectangle "CombRect1" {height = 2 : ui32, width = 1 : ui32, xCoord = 0 : i64, yCoord = 0 : i64}
    chalk.rectangle "CombRect2" {height = 2 : ui32, width = 2 : ui32, xCoord = 0 : i64, yCoord = 0 : i64}
    chalk.rectangle "CombRect3" {height = 2 : ui32, width = 3 : ui32, xCoord = 0 : i64, yCoord = 0 : i64}
}
