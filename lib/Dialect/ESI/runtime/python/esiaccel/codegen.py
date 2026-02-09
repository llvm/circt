#  Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
#  See https://llvm.org/LICENSE.txt for license information.
#  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Code generation from ESI manifests to source code.

Uses a two-pass approach for C++: first collect and name all reachable types,
then emit structs/aliases in a dependency-ordered sequence so headers are
standalone and deterministic.
"""

# C++ header support included with the runtime, though it is intended to be
# extensible for other languages.

from typing import Dict, List, Set, TextIO, Tuple, Type, Optional
from .accelerator import AcceleratorConnection, Context
from .esiCppAccel import ModuleInfo
from . import esiCppAccel as cpp
from . import types

import argparse
from pathlib import Path
import textwrap
import sys

_thisdir = Path(__file__).absolute().resolve().parent


class Generator:
  """Base class for all generators."""

  language: Optional[str] = None

  def __init__(self, conn: AcceleratorConnection):
    self.manifest = conn.manifest()

  def generate(self, output_dir: Path, system_name: str):
    raise NotImplementedError("Generator.generate() must be overridden")


class CppGenerator(Generator):
  """Generate C++ headers from an ESI manifest."""

  language = "C++"

  # Supported bit widths for lone integer types.
  int_width_support = set([1, 8, 16, 32, 64])

  def get_type_str(self, type: types.ESIType) -> str:
    """Get the textual code for the storage class of a type.

    Examples: uint32_t, int64_t, CustomStruct."""

    if isinstance(type, (types.BitsType, types.IntType)):
      if type.bit_width == 1:
        return "bool"

      if type.bit_width <= 8:
        storage_width = 8
      elif type.bit_width <= 16:
        storage_width = 16
      elif type.bit_width <= 32:
        storage_width = 32
      elif type.bit_width <= 64:
        storage_width = 64
      else:
        raise ValueError(f"Unsupported integer width: {type.bit_width}")

      if isinstance(type, (types.BitsType, types.UIntType)):
        return f"uint{storage_width}_t"
      return f"int{storage_width}_t"
    raise NotImplementedError(f"Type '{type}' not supported for C++ generation")

  def get_consts_str(self, module_info: ModuleInfo) -> str:
    """Get the C++ code for a constant in a module."""
    const_strs: List[str] = [
        f"static constexpr {self.get_type_str(const.type)} "
        f"{name} = 0x{const.value:x};"
        for name, const in module_info.constants.items()
    ]
    return "\n".join(const_strs)

  def write_modules(self, output_dir: Path, system_name: str):
    """Write the C++ header. One for each module in the manifest."""

    for module_info in self.manifest.module_infos:
      s = f"""
      /// Generated header for {system_name} module {module_info.name}.
      #pragma once
      #include "types.h"

      namespace {system_name} {{
      class {module_info.name} {{
      public:
        {self.get_consts_str(module_info)}
      }};
      }} // namespace {system_name}
      """

      hdr_file = output_dir / f"{module_info.name}.h"
      with open(hdr_file, "w") as hdr:
        hdr.write(textwrap.dedent(s))

  def write_type(self, hdr: TextIO, type: types.ESIType):
    if isinstance(type, (types.BitsType, types.IntType)):
      # Bit vector types use standard C++ types.
      return
    raise NotImplementedError(f"Type '{type}' not supported for C++ generation")

  def write_types(self, output_dir: Path, system_name: str):
    """Generate types.h with struct/alias emission and dependency ordering."""
    type_id_map: Dict[str, str] = {}
    type_by_id: Dict[str, types.ESIType] = {}
    alias_name_by_id: Dict[str, str] = {}
    struct_name_by_id: Dict[str, str] = {}
    used_names: Set[str] = set()

    def get_type_id(t: object) -> str:
      if isinstance(t, types.ESIType):
        return t.cpp_type.id
      return t.id

    def wrap_type(t: object) -> types.ESIType:
      if isinstance(t, types.ESIType):
        return t
      return types._get_esi_type(t)

    def sanitize_name(name: str) -> str:
      """Map a type or alias id into a C++-friendly identifier."""
      name = name.replace("::", "_")
      if name.startswith("@"):
        name = name[1:]
      sanitized = []
      for ch in name:
        if ch.isalnum() or ch == "_":
          sanitized.append(ch)
        else:
          sanitized.append("_")
      if not sanitized:
        return "Type"
      if sanitized[0].isdigit():
        sanitized.insert(0, "_")
      return "".join(sanitized)

    def reserve_name(base: str) -> str:
      """Reserve a unique name, appending a numeric suffix if needed."""
      base = sanitize_name(base)
      name = base
      idx = 1
      while name in used_names:
        name = f"{base}_{idx}"
        idx += 1
      used_names.add(name)
      return name

    def auto_struct_name(struct_type: types.StructType) -> str:
      """Auto-name anonymous structs from their field names and type ids."""
      parts = ["_struct"]
      for field_name, field_type in struct_type.fields:
        parts.append(field_name)
        parts.append(sanitize_name(get_type_id(field_type)))
      return reserve_name("_".join(parts))

    def get_alias_info(
        wrapped: types.ESIType) -> Optional[Tuple[str, Optional[types.ESIType]]]:
      """Return (alias_name, inner_type) if this is a TypeAlias."""
      if isinstance(wrapped, types.TypeAlias):
        return (wrapped.name, wrapped.inner_type)
      return None

    def is_opaque_any(wrapped: types.ESIType) -> bool:
      """Detect legacy any-type ids which still surface as raw ids."""
      return get_type_id(wrapped) == "!esi.any"

    def register_type(t: object) -> None:
      """Collect all reachable types via bundles, channels, structs, arrays."""
      wrapped = wrap_type(t)
      tid = get_type_id(wrapped)
      if tid in type_by_id:
        return
      type_by_id[tid] = wrapped
      if isinstance(wrapped.cpp_type, cpp.BundleType):
        for _, _, chan_type in wrapped.cpp_type.channels:
          register_type(chan_type)
        return
      if isinstance(wrapped.cpp_type, cpp.ChannelType):
        register_type(wrapped.cpp_type.inner)
        return
      if isinstance(wrapped, types.StructType):
        for _, field_type in wrapped.fields:
          register_type(field_type)
        return
      if isinstance(wrapped, types.ArrayType):
        register_type(wrapped.element_type)
        return

    for t in self.manifest.type_table:
      register_type(t)

    for tid, t in type_by_id.items():
      alias_info = get_alias_info(t)
      if alias_info is None:
        continue
      alias_raw, inner_wrapped = alias_info
      alias_name = alias_name_by_id.get(tid)
      if alias_name is None:
        alias_name = reserve_name(alias_raw)
        alias_name_by_id[tid] = alias_name
      if inner_wrapped is not None and isinstance(inner_wrapped,
                                                  types.StructType):
        struct_name_by_id.setdefault(get_type_id(inner_wrapped), alias_name)

    aliased_structs: Set[str] = set()
    # Track structs named via aliases to prioritize their emission order.
    for tid, t in type_by_id.items():
      alias_info = get_alias_info(t)
      if alias_info is None:
        continue
      _, inner_wrapped = alias_info
      if inner_wrapped is not None and isinstance(inner_wrapped,
                                                  types.StructType):
        aliased_structs.add(get_type_id(inner_wrapped))

    for tid, t in type_by_id.items():
      if isinstance(wrap_type(t), types.StructType):
        if tid not in struct_name_by_id:
          struct_name_by_id[tid] = auto_struct_name(wrap_type(t))

    def format_array_decl(array_type: types.ArrayType, name: str) -> str:
      """Emit a C++ field declaration for a possibly multi-dim array."""
      dims: List[int] = []
      inner: types.ESIType = array_type
      while isinstance(inner, types.ArrayType):
        dims.append(inner.size)
        inner = inner.element_type
      base_cpp = type_id_map[get_type_id(inner)]
      suffix = "".join([f"[{d}]" for d in dims])
      return f"{base_cpp} {name}{suffix};"

    def format_array_type(array_type: types.ArrayType) -> str:
      """Emit a C++ array type for aliases and type table mapping."""
      dims: List[int] = []
      inner: types.ESIType = array_type
      while isinstance(inner, types.ArrayType):
        dims.append(inner.size)
        inner = inner.element_type
      base_cpp = type_id_map[get_type_id(inner)]
      suffix = "".join([f"[{d}]" for d in dims])
      return f"{base_cpp}{suffix}"

    def assign_type_name(t: object) -> None:
      """Assign a C++ name for each type id, preserving alias intent."""
      wrapped = wrap_type(t)
      tid = get_type_id(wrapped)
      if tid in type_id_map:
        return

      alias_info = get_alias_info(wrapped)
      if alias_info is not None:
        alias_name, inner_wrapped = alias_info
        if tid not in alias_name_by_id:
          alias_name_by_id[tid] = reserve_name(alias_name)
        alias_name = alias_name_by_id[tid]
        if inner_wrapped is not None:
          if isinstance(inner_wrapped, types.StructType):
            inner_tid = get_type_id(inner_wrapped)
            struct_name_by_id[inner_tid] = alias_name
            type_id_map[inner_tid] = alias_name
            type_id_map[tid] = alias_name
            return
          assign_type_name(inner_wrapped)
        type_id_map[tid] = alias_name
        return

      if isinstance(wrapped.cpp_type, cpp.BundleType):
        for _, _, chan_type in wrapped.cpp_type.channels:
          assign_type_name(chan_type)
        type_id_map[tid] = "void"
        return

      if isinstance(wrapped.cpp_type, cpp.ChannelType):
        assign_type_name(wrapped.cpp_type.inner)
        type_id_map[tid] = type_id_map[get_type_id(wrapped.cpp_type.inner)]
        return

      if isinstance(wrapped, types.VoidType):
        type_id_map[tid] = "void"
        return

      if isinstance(wrapped, types.AnyType):
        type_id_map[tid] = "std::any"
        return

      if is_opaque_any(wrapped):
        type_id_map[tid] = "std::any"
        return

      if isinstance(wrapped, (types.BitsType, types.IntType)):
        type_id_map[tid] = self.get_type_str(wrapped)
        return

      if isinstance(wrapped, types.ArrayType):
        assign_type_name(wrapped.element_type)
        type_id_map[tid] = format_array_type(wrapped)
        return

      if isinstance(wrapped, types.StructType):
        for _, field_type in wrapped.fields:
          assign_type_name(field_type)
        type_id_map[tid] = struct_name_by_id[tid]
        return

      if type(wrapped) is types.ESIType:
        type_id_map[tid] = "std::any"
        return

      raise NotImplementedError(
          f"Type '{wrapped}' not supported for C++ generation")

    for esi_type in self.manifest.type_table:
      assign_type_name(esi_type)

    emit_nodes: Dict[Tuple[str, str], types.ESIType] = {}
    emit_deps: Dict[Tuple[str, str], Set[Tuple[str, str]]] = {}
    emit_index: Dict[Tuple[str, str], int] = {}
    emit_counter = 0

    def emit_struct(struct_type: types.StructType) -> None:
      """Emit a packed C++ struct with a stable type id field."""
      tid = get_type_id(struct_type)

      fields = list(struct_type.fields)
      if struct_type.cpp_type.reverse:
        fields = list(reversed(fields))

      field_decls: List[str] = []
      for field_name, field_type in fields:
        field_wrapped = wrap_type(field_type)
        if isinstance(field_wrapped, types.ArrayType):
          field_decls.append(format_array_decl(field_wrapped, field_name))
        else:
          field_cpp = type_id_map[get_type_id(field_wrapped)]
          field_decls.append(f"{field_cpp} {field_name};")

      hdr.write(f"struct {type_id_map[tid]} {{\n")
      for decl in field_decls:
        hdr.write(f"  {decl}\n")
      hdr.write("\n")
      hdr.write(
          f"  static constexpr std::string_view id = \"{get_type_id(struct_type)}\";\n"
      )
      hdr.write("};\n\n")
    def dep_nodes_for_type(wrapped: types.ESIType) -> Set[Tuple[str, str]]:
      """Return dependency nodes for topo ordering (struct/alias only)."""
      if isinstance(wrapped, types.StructType):
        return {("struct", get_type_id(wrapped))}

      if isinstance(wrapped, types.TypeAlias):
        inner = wrapped.inner_type
        if isinstance(inner, types.StructType):
          return {("struct", get_type_id(inner))}
        return {("alias", get_type_id(wrapped))}

      if isinstance(wrapped, types.ArrayType):
        return dep_nodes_for_type(wrapped.element_type)

      return set()

    def add_node(kind: str, wrapped: types.ESIType) -> Tuple[str, str]:
      """Register an emission node, keeping insertion order for tie-breaks."""
      nonlocal emit_counter
      tid = get_type_id(wrapped)
      key = (kind, tid)
      if key not in emit_nodes:
        emit_nodes[key] = wrapped
        emit_deps[key] = set()
        emit_index[key] = emit_counter
        emit_counter += 1
      return key

    def visit_type(t: object) -> None:
      """Populate nodes and dependencies for structs and aliases."""
      wrapped = wrap_type(t)
      tid = get_type_id(wrapped)

      alias_info = get_alias_info(wrapped)
      if alias_info is not None:
        alias_name, inner_wrapped = alias_info
        alias_name = alias_name_by_id.get(tid, alias_name)
        if inner_wrapped is not None and isinstance(inner_wrapped,
                                                    types.StructType):
          visit_type(inner_wrapped)
          return
        if inner_wrapped is not None:
          visit_type(inner_wrapped)
        key = add_node("alias", wrapped)
        if inner_wrapped is not None:
          emit_deps[key].update(dep_nodes_for_type(inner_wrapped))
        return

      if isinstance(wrapped.cpp_type, cpp.BundleType):
        for _, _, chan_type in wrapped.cpp_type.channels:
          visit_type(chan_type)
        return

      if isinstance(wrapped.cpp_type, cpp.ChannelType):
        visit_type(wrapped.cpp_type.inner)
        return

      if isinstance(wrapped, types.StructType):
        key = add_node("struct", wrapped)
        for _, field_type in wrapped.fields:
          visit_type(field_type)
          emit_deps[key].update(dep_nodes_for_type(wrap_type(field_type)))
        emit_deps[key].discard(key)
        return

      if isinstance(wrapped, types.ArrayType):
        visit_type(wrapped.element_type)
        return

      if is_opaque_any(wrapped):
        return

      if isinstance(wrapped, types.AnyType):
        return

    def emit_alias(alias_type: types.ESIType) -> None:
      """Emit a using-alias when the alias maps to a distinct C++ type."""
      tid = get_type_id(alias_type)
      alias_info = get_alias_info(alias_type)
      if alias_info is None:
        return
      alias_name, inner_wrapped = alias_info
      alias_name = alias_name_by_id.get(tid, alias_name)
      inner_cpp = None
      if inner_wrapped is not None:
        inner_cpp = type_id_map[get_type_id(inner_wrapped)]
      if inner_cpp is None:
        inner_cpp = type_id_map[tid]
      if inner_cpp != alias_name:
        hdr.write(f"using {alias_name} = {inner_cpp};\n\n")

    hdr_file = output_dir / "types.h"
    with open(hdr_file, "w") as hdr:
      hdr.write(
          textwrap.dedent(f"""
      // Generated header for {system_name} types.
      #pragma once

      #include <cstdint>
      #include <any>
      #include <string_view>

      namespace {system_name} {{
      #pragma pack(push, 1)
      """))

      for esi_type in self.manifest.type_table:
        try:
          visit_type(esi_type)
        except NotImplementedError:
          sys.stderr.write(
              f"Warning: type '{esi_type}' not supported for C++ generation\n")

      indegree: Dict[Tuple[str, str], int] = {}
      for key in emit_nodes:
        indegree[key] = 0
      for key, deps in emit_deps.items():
        for dep in deps:
          if dep in indegree:
            indegree[key] += 1

      ready = [key for key, deg in indegree.items() if deg == 0]
      emitted = 0
      # Emit in dependency order, preferring aliases and alias-named structs.
      while ready:
        def sort_name(k: Tuple[str, str]) -> str:
          kind, tid = k
          if kind == "alias":
            return alias_name_by_id.get(tid, tid)
          if kind == "struct":
            return struct_name_by_id.get(tid, tid)
          return tid

        ready.sort(
            key=lambda k: (0 if k[0] == "alias" or
                           (k[0] == "struct" and k[1] in aliased_structs) else
                           1, sort_name(k), emit_index[k]))
        key = ready.pop(0)
        kind, _ = key
        wrapped = emit_nodes[key]
        if kind == "struct":
          emit_struct(wrapped)
        elif kind == "alias":
          emit_alias(wrapped)
        emitted += 1
        for other_key, deps in emit_deps.items():
          if key in deps:
            deps.remove(key)
            indegree[other_key] -= 1
            if indegree[other_key] == 0:
              ready.append(other_key)

      if emitted != len(emit_nodes):
        sys.stderr.write("Warning: cyclic type dependencies detected\n")

      hdr.write(
          textwrap.dedent(f"""
      #pragma pack(pop)
      }} // namespace {system_name}
      """))

    self.type_id_map = type_id_map

  def generate(self, output_dir: Path, system_name: str):
    self.write_types(output_dir, system_name)
    self.write_modules(output_dir, system_name)


def run(generator: Type[Generator] = CppGenerator,
        cmdline_args=sys.argv) -> int:
  """Create and run a generator reading options from the command line."""

  argparser = argparse.ArgumentParser(
      description=f"Generate {generator.language} headers from an ESI manifest",
      formatter_class=argparse.RawDescriptionHelpFormatter,
      epilog=textwrap.dedent("""
        Can read the manifest from either a file OR a running accelerator.

        Usage examples:
          # To read the manifest from a file:
          esi-cppgen --file /path/to/manifest.json

          # To read the manifest from a running accelerator:
          esi-cppgen --platform cosim --connection localhost:1234
      """))

  argparser.add_argument("--file",
                         type=str,
                         default=None,
                         help="Path to the manifest file.")
  argparser.add_argument(
      "--platform",
      type=str,
      help="Name of platform for live accelerator connection.")
  argparser.add_argument(
      "--connection",
      type=str,
      help="Connection string for live accelerator connection.")
  argparser.add_argument(
      "--output-dir",
      type=str,
      default="esi",
      help="Output directory for generated files. Recommend adding either `esi`"
      " or the system name to the end of the path so as to avoid header name"
      "conflicts. Defaults to `esi`")
  argparser.add_argument(
      "--system-name",
      type=str,
      default="esi_system",
      help="Name of the ESI system. For C++, this will be the namespace.")

  if (len(cmdline_args) <= 1):
    argparser.print_help()
    return 1
  args = argparser.parse_args(cmdline_args[1:])

  if args.file is not None and args.platform is not None:
    print("Cannot specify both --file and --platform")
    return 1

  conn: AcceleratorConnection
  if args.file is not None:
    conn = Context.default().connect("trace", f"-:{args.file}")
  elif args.platform is not None:
    if args.connection is None:
      print("Must specify --connection with --platform")
      return 1
    conn = Context.default().connect(args.platform, args.connection)
  else:
    print("Must specify either --file or --platform")
    return 1

  output_dir = Path(args.output_dir)
  if output_dir.exists() and not output_dir.is_dir():
    print(f"Output directory {output_dir} is not a directory")
    return 1
  if not output_dir.exists():
    output_dir.mkdir(parents=True)

  gen = generator(conn)
  gen.generate(output_dir, args.system_name)
  return 0


if __name__ == '__main__':
  sys.exit(run())
