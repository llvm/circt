#  Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
#  See https://llvm.org/LICENSE.txt for license information.
#  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import inspect
import argparse
import importlib.util
import sys
from pathlib import Path
from enum import Enum
import os
from types import ModuleType

import pyrtg
from pyrtg.circt import ir, passmanager, register_dialects


class InputFormat(Enum):
  """
  The input formats accepted by this tool. The enum's values correspond to the
  selectable CLI argument choice, but not necessarily the file extension.
  """

  MLIR = 'mlir'
  PYTHON = 'py'


class OutputFormat(Enum):
  """
  The output formats accepted by this tool. The enum's values correspond to the
  selectable CLI argument choice, but not necessarily the file extension.
  """

  MLIR = 'mlir'
  ELABORATED = 'elaborated'
  ASM = 'asm'


def parse_args() -> argparse.Namespace:
  """
  Parse and post-process the tool's CLI arguments.
  """

  output_format_choices = ["mlir", "elaborated", "asm"]

  # Set up argument parser
  parser = argparse.ArgumentParser(
      description="CIRCT Random Test Generation (RTG) Driver")
  parser.add_argument("file",
                      type=str,
                      help="Path to the Python file to import")
  parser.add_argument("--seed",
                      type=int,
                      required=True,
                      help="Seed for all RNGs during randomization")
  parser.add_argument("--verify-passes",
                      type=bool,
                      default=True,
                      help="Run the verifier after each transformation pass")
  parser.add_argument("--input-format",
                      type=InputFormat,
                      choices=list(InputFormat),
                      help="Input Format")
  parser.add_argument("--output-format",
                      type=OutputFormat,
                      choices=list(OutputFormat),
                      default=OutputFormat.ASM,
                      help="Output Format")
  parser.add_argument(
      "-o",
      "--output-path",
      type=str,
      default="-",
      help=
      "Output Path (directory if 'split-output=True' or a filepath otherwise)")
  parser.add_argument("--memories-as-immediates",
                      type=bool,
                      default=True,
                      help="Lower memories as immediates")
  parser.add_argument(
      "--mlir-timing",
      type=bool,
      nargs='?',
      default=False,
      const=True,
      help="Print pass timings of the MLIR compilation pipeline")

  args = parser.parse_args()

  # Convert to an absolute path because the backend does not support relative paths.
  if args.output_path not in ["", "-"]:
    args.output_path = os.path.abspath(args.output_path)

  return args


def import_module_from_path(module_path: Path) -> ModuleType:
  """
  Given the file path of an input Python file, import it as a Python module and return it.
  """

  module_name = module_path.stem  # Removes the '.py' part

  # Load the module dynamically
  spec = importlib.util.spec_from_file_location(module_name, module_path)
  module = importlib.util.module_from_spec(spec)
  sys.modules[module_name] = module
  spec.loader.exec_module(module)

  return module


def frontend_codegen(args: argparse.Namespace) -> ir.Module:
  """
  Given the user arguments, generate the initial MLIR module for further
  processing.
  """

  # If we get MLIR directly, just read the file and parse the MLIR
  if args.input_format == InputFormat.MLIR:
    with open(args.file, 'r') as f:
      return ir.Module.parse(f.read())

  # Otherwise, codegen the MLIR from the python file
  if args.input_format == InputFormat.PYTHON:
    import_module_from_path(Path(args.file))

    module = ir.Module.create(loc=ir.Location.file(args.file, 0, 0))
    with ir.InsertionPoint(module.body):
      pyrtg.core.CodeGenRoot._codegen_all_instances()
    return module

  assert False, "input format must be one of the above"


def compile(mlir_module: ir.Module, args: argparse.Namespace) -> None:
  """
  Populate and run the MLIR compilation pipeline according to the CLI arguments.
  """

  def get_populated_pm():
    pm = passmanager.PassManager()
    pm.enable_verifier(args.verify_passes)
    if args.mlir_timing:
      pm.enable_timing()
    pm.add('any(cse,canonicalize)')

    if args.output_format == OutputFormat.MLIR:
      return pm

    pm.add(
        f'rtg-randomization-pipeline{{seed={args.seed} memories-as-immediates={str(args.memories_as_immediates).lower()}}}'
    )

    if args.output_format == OutputFormat.ELABORATED:
      return pm

    pm.add(
        f'rtg-insert-test-to-file-mapping{{split-output=false path={args.output_path}}}'
    )
    pm.add('rtg-simple-test-inliner')
    pm.add('emit.file(rtg-emit-isa-assembly)')
    return pm

  try:
    get_populated_pm().run(mlir_module.operation)
  except Exception as e:
    print(str(e), file=sys.stderr)
    sys.exit(1)


def print_output(mlir_module: ir.Module, args: argparse.Namespace) -> None:
  """
  Print the tool output according to the CLI arguments.
  """

  # The assembly emitter does print to the desired output itself, so we don't need to do anything here.
  if args.output_format == OutputFormat.ASM:
    return

  if len(args.output_path) == 0:
    mlir_module.operation.print(file=sys.stderr, enable_debug_info=True)
  elif args.output_path == "-":
    mlir_module.operation.print(enable_debug_info=True)
  else:
    with open(args.output_path, 'w') as f:
      mlir_module.operation.print(file=f, enable_debug_info=True)


def infer_input_format(filename: str) -> InputFormat | None:
  """
  If the input format was not explicitly provided by the user, this function can
  try to infer it based on the file name.
  """

  _, file_extension = os.path.splitext(filename)

  if file_extension == ".mlir":
    return InputFormat.MLIR

  if file_extension == ".py":
    return InputFormat.PYTHON


def run() -> None:
  """
  The rtgtool entry point.
  """

  args = parse_args()

  # Create the MLIR context and register all CIRCT dialects. In principle, we'd
  # only need to register the RTG and the desired payload dialects.
  with ir.Context() as ctx, ir.Location.unknown():
    register_dialects(ctx)

    # If no input format is given, try to infer it from the file extension
    if args.input_format == None:
      args.input_format = infer_input_format(args.file)

    # If we still don't know how to process the input file, return an error.
    if args.input_format == None:
      print(
          "Input format not provided and cannot be inferred automatically. Please use '--input-format' to specify the input format.",
          file=sys.stderr)
      return

    # Produce the MLIR from the frontend language (which might already be MLIR)
    mlir_module = frontend_codegen(args)
    if not mlir_module.operation.verify():
      return

    # Compile the MLIR up to the specified point.
    compile(mlir_module, args)

    # Print the output or skip it if the desired emission was already part of the lowering pipeline.
    print_output(mlir_module, args)


if __name__ == "__main__":
  run()
