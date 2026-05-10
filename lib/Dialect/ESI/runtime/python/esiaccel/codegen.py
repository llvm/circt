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
from . import types
from .types import (BundlePort as _BundlePort, FunctionPort as _FunctionPort,
                    CallbackPort as _CallbackPort, ToHostPort as _ToHostPort,
                    FromHostPort as _FromHostPort, MMIORegion as
                    _MMIORegionPort, MetricPort as _MetricPort)

import sys
import os
import textwrap
import argparse
from dataclasses import dataclass, field as _dc_field
from pathlib import Path

_thisdir = Path(__file__).absolute().resolve().parent


@dataclass
class _PortGroup:
  """All strings needed to emit one port slot (member + ctor + find) in Connected."""
  struct_decls: List[str] = _dc_field(default_factory=list)
  member_decl: str = ""
  ctor_params: List[str] = _dc_field(default_factory=list)
  init_entry: str = ""
  find_code: str = ""
  make_unique_args: List[str] = _dc_field(default_factory=list)
  post_connect: str = ""
  using_aliases: List[Tuple[str, str]] = _dc_field(default_factory=list)


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

  def __init__(self, conn: AcceleratorConnection):
    super().__init__(conn)
    self._conn = conn
    self.type_planner = CppTypePlanner(self.manifest.type_table)
    self.type_emitter = CppTypeEmitter(self.type_planner)

  def get_consts_str(self, module_info: ModuleInfo) -> str:
    """Get the C++ code for a constant in a module."""
    const_strs: List[str] = [
        f"static constexpr {self.type_emitter.type_identifier(const.type)} "
        f"{name} = 0x{const.value:x};"
        for name, const in module_info.constants.items()
    ]
    return "\n".join(const_strs)

  # ---------------------------------------------------------------------------
  # Port-emission helpers
  # ---------------------------------------------------------------------------

  @staticmethod
  def _sanitize_id(name: str) -> str:
    """Return a C++-safe identifier from an AppID name."""
    result = []
    for ch in name:
      result.append(ch if (ch.isalnum() or ch == "_") else "_")
    if not result:
      return "_port"
    if result[0].isdigit():
      result.insert(0, "_")
    return "".join(result)

  def _build_module_instance_map(self) -> Dict[str, object]:
    """Walk the live hierarchy and return {module_name: first Instance}."""
    accel = self._conn.build_accelerator()
    result: Dict[str, object] = {}
    queue = list(accel.children.values())
    while queue:
      inst = queue.pop(0)
      info = inst.cpp_hwmodule.info
      if info is not None:
        name = info.name
        if name is not None and name not in result:
          result[name] = inst
      queue.extend(inst.children.values())
    return result

  def _cpp_member_type(self, port, alias_prefix: Optional[str] = None) -> str:
    """Return the C++ member type string for a port (no member name).

    For typed ports (function/callback/to-host/from-host channels) `alias_prefix`
    is required; the returned template parameters are written using the alias
    names (`<prefix>Args`, `<prefix>Result`, `<prefix>Data`) that should be
    emitted at module-class scope via `_port_using_aliases`. For non-typed
    ports (MMIO regions, telemetry metrics, plain bundles) `alias_prefix` is
    ignored and the runtime reference/pointer type is returned directly.
    """
    if isinstance(port, _FunctionPort):
      assert alias_prefix is not None, (
          "alias_prefix is required for FunctionPort to avoid emitting "
          "long mangled type names inline (which would also collide across "
          "modules' `using` declarations)")
      return (f"esi::TypedFunction<{alias_prefix}Args, "
              f"{alias_prefix}Result>")
    if isinstance(port, _CallbackPort):
      assert alias_prefix is not None, (
          "alias_prefix is required for CallbackPort")
      return (f"esi::TypedCallback<{alias_prefix}Args, "
              f"{alias_prefix}Result>")
    if isinstance(port, _ToHostPort):
      assert alias_prefix is not None, (
          "alias_prefix is required for ToHostPort")
      return f"esi::TypedReadPort<{alias_prefix}Data>"
    if isinstance(port, _FromHostPort):
      assert alias_prefix is not None, (
          "alias_prefix is required for FromHostPort")
      return f"esi::TypedWritePort<{alias_prefix}Data>"
    if isinstance(port, _MMIORegionPort):
      return "esi::services::MMIO::MMIORegion &"
    if isinstance(port, _MetricPort):
      return "esi::services::TelemetryService::Metric &"
    return "esi::BundlePort &"

  def _port_using_aliases(self, alias_prefix: str,
                          port) -> List[Tuple[str, str]]:
    """Return (alias_name, type_id) pairs to emit as `using` declarations at
    module-class scope for the typed-port template parameters."""
    if isinstance(port, (_FunctionPort, _CallbackPort)):
      arg = self.type_emitter.type_identifier(port.arg_window_type or
                                              port.arg_type)
      res = self.type_emitter.type_identifier(port.result_window_type or
                                              port.result_type)
      return [(f"{alias_prefix}Args", arg), (f"{alias_prefix}Result", res)]
    if isinstance(port, (_ToHostPort, _FromHostPort)):
      data = self.type_emitter.type_identifier(port.data_window_type or
                                               port.data_type)
      return [(f"{alias_prefix}Data", data)]
    return []

  def _cpp_indexed_elem_type(self,
                             port,
                             alias_prefix: Optional[str] = None) -> str:
    """Return the storage type used inside an `IndexedPorts<T>` for `port`.

    Typed ports use the same `TypedFunction<...>` / `TypedReadPort<...>` etc.
    that `_cpp_member_type` produces. MMIO regions, telemetry metrics, and
    plain bundle ports are stored as raw pointers because `std::map<int, T&>`
    is ill-formed.
    """
    if isinstance(port, _MMIORegionPort):
      return "esi::services::MMIO::MMIORegion *"
    if isinstance(port, _MetricPort):
      return "esi::services::TelemetryService::Metric *"
    if isinstance(port,
                  (_FunctionPort, _CallbackPort, _ToHostPort, _FromHostPort)):
      return self._cpp_member_type(port, alias_prefix=alias_prefix)
    return "esi::BundlePort *"

  def _cpp_ctor_param_type(self, port) -> str:
    """Return the C++ constructor parameter type for a port."""
    if isinstance(port, _FunctionPort):
      return "esi::services::FuncService::Function *"
    if isinstance(port, _CallbackPort):
      return "esi::services::CallService::Callback *"
    if isinstance(port, _ToHostPort):
      return "esi::ReadChannelPort &"
    if isinstance(port, _FromHostPort):
      return "esi::WriteChannelPort &"
    if isinstance(port, _MMIORegionPort):
      return "esi::services::MMIO::MMIORegion &"
    if isinstance(port, _MetricPort):
      return "esi::services::TelemetryService::Metric &"
    return "esi::BundlePort &"

  @staticmethod
  def _cpp_ctor_param_suffix(port) -> str:
    """Return the parameter name suffix ('_chan', '_svc', or '_port')."""
    if isinstance(port, (_ToHostPort, _FromHostPort)):
      return "_chan"
    if isinstance(port, (_MMIORegionPort, _MetricPort)):
      return "_svc"
    return "_port"

  @staticmethod
  def _appid_expr(appid) -> str:
    """Return `esi::AppID(...)` expression for an AppID."""
    name = appid.name
    idx = appid.idx
    if idx is None:
      return f'esi::AppID("{name}")'
    return f'esi::AppID("{name}", {idx})'

  def _port_find_code(self, member_name: str, port, appid) -> str:
    """Return the code snippet that resolves a scalar port in connect()."""
    ae = self._appid_expr(appid)
    if isinstance(port, _FunctionPort):
      v = f"{member_name}_port"
      return (
          f"auto *{v} =\n"
          f"    esi::findPortAsOrThrow<esi::services::FuncService::Function>(\n"
          f"        rawModule, {ae});")
    if isinstance(port, _CallbackPort):
      v = f"{member_name}_port"
      return (
          f"auto *{v} =\n"
          f"    esi::findPortAsOrThrow<esi::services::CallService::Callback>(\n"
          f"        rawModule, {ae});")
    if isinstance(port, _ToHostPort):
      v = f"{member_name}_chan"
      return (
          f"auto &{v} =\n"
          f"    esi::findPortAsOrThrow<esi::services::ChannelService::ToHost>(\n"
          f'        rawModule, {ae})->getRawRead("data");')
    if isinstance(port, _FromHostPort):
      v = f"{member_name}_chan"
      return (
          f"auto &{v} =\n"
          f"    esi::findPortAsOrThrow<esi::services::ChannelService::FromHost>(\n"
          f'        rawModule, {ae})->getRawWrite("data");')
    if isinstance(port, _MMIORegionPort):
      v = f"{member_name}_svc"
      return (f"auto &{v} =\n"
              f"    *esi::findPortAsOrThrow<esi::services::MMIO::MMIORegion>(\n"
              f"        rawModule, {ae});")
    if isinstance(port, _MetricPort):
      v = f"{member_name}_svc"
      return (
          f"auto &{v} =\n"
          f"    *esi::findPortAsOrThrow<esi::services::TelemetryService::Metric>(\n"
          f"        rawModule, {ae});")
    # plain BundlePort fallback
    v = f"{member_name}_port"
    return f"auto &{v} = esi::findPortOrThrow(rawModule, {ae});"

  @staticmethod
  def _port_make_unique_arg(member_name: str, port) -> str:
    """Return the argument expression for make_unique<Connected>(...)."""
    if isinstance(port, (_ToHostPort, _FromHostPort)):
      return f"{member_name}_chan"
    if isinstance(port, (_MMIORegionPort, _MetricPort)):
      return f"{member_name}_svc"
    return f"{member_name}_port"

  @staticmethod
  def _port_is_connectable(port) -> bool:
    """True if the generated connect() should call .connect() on this port."""
    return isinstance(port, (_FunctionPort, _ToHostPort, _FromHostPort))
    # CallbackPort.connect() requires a user-supplied callback — skip.
    # MMIORegion, Metric, BundlePort — no .connect() method.

  def _scalar_port_group(self, member_name: str, port, appid) -> _PortGroup:
    """Build a _PortGroup for a single scalar (non-indexed) port."""
    aliases = self._port_using_aliases(member_name, port)
    alias_prefix = member_name if aliases else None
    member_type = self._cpp_member_type(port, alias_prefix=alias_prefix)
    is_ref = member_type.endswith(" &")
    param_type = self._cpp_ctor_param_type(port)
    param_suffix = self._cpp_ctor_param_suffix(port)
    param_name = f"{member_name}{param_suffix}"

    if is_ref:
      member_decl = f"{member_type}{member_name};"
    else:
      member_decl = f"{member_type} {member_name};"

    post = ""
    if self._port_is_connectable(port):
      post = f"connected->{member_name}.connect();"

    return _PortGroup(
        member_decl=member_decl,
        ctor_params=[f"{param_type} {param_name}"],
        init_entry=f"{member_name}({param_name})",
        find_code=self._port_find_code(member_name, port, appid),
        make_unique_args=[self._port_make_unique_arg(member_name, port)],
        post_connect=post,
        using_aliases=aliases,
    )

  def _indexed_ports_group(self, member_name: str, appid_name: str,
                           port_list) -> _PortGroup:
    """Build a _PortGroup for a same-name, same-type indexed port array."""
    # Derive the element type from the first port.
    first_port = port_list[0][1]
    aliases = self._port_using_aliases(member_name, first_port)
    alias_prefix = member_name if aliases else None
    elem_type = self._cpp_indexed_elem_type(first_port,
                                            alias_prefix=alias_prefix)
    indexed_type = f"esi::IndexedPorts<{elem_type}>"
    map_var = f"{member_name}_backing"
    map_type = f"std::map<int, {elem_type}>"
    indexed_var = f"{member_name}_map"

    # Build the find code: per-index resolve and try_emplace, then freeze
    # into the IndexedPorts wrapper. The body of the loop differs by port
    # kind: channel ports need an extra `getRawRead("data")` /
    # `getRawWrite("data")` step, MMIO regions and metrics store raw
    # pointers.
    find_parts = [
        f"{map_type} {map_var};",
        f"for (uint32_t idx : esi::findPortIndices(rawModule, "
        f"\"{appid_name}\")) {{",
    ]
    appid_expr = f'esi::AppID("{appid_name}", idx)'
    if isinstance(first_port, _FunctionPort):
      find_parts.append(
          f"  {map_var}.try_emplace(\n"
          f"      static_cast<int>(idx),\n"
          f"      esi::findPortAsOrThrow<esi::services::FuncService::Function>"
          f"(\n"
          f"          rawModule, {appid_expr}));")
    elif isinstance(first_port, _CallbackPort):
      find_parts.append(
          f"  {map_var}.try_emplace(\n"
          f"      static_cast<int>(idx),\n"
          f"      esi::findPortAsOrThrow<esi::services::CallService::Callback>"
          f"(\n"
          f"          rawModule, {appid_expr}));")
    elif isinstance(first_port, _ToHostPort):
      # TypedReadPort takes a ReadChannelPort&, not the service port. Resolve
      # the service port first, then bind its underlying raw read channel.
      find_parts.append(
          f"  auto *svc =\n"
          f"      esi::findPortAsOrThrow<esi::services::ChannelService::ToHost>"
          f"(\n"
          f"          rawModule, {appid_expr});\n"
          f"  {map_var}.try_emplace(\n"
          f"      static_cast<int>(idx),\n"
          f"      svc->getRawRead(\"data\"));")
    elif isinstance(first_port, _FromHostPort):
      find_parts.append(
          f"  auto *svc =\n"
          f"      esi::findPortAsOrThrow<esi::services::ChannelService::"
          f"FromHost>(\n"
          f"          rawModule, {appid_expr});\n"
          f"  {map_var}.try_emplace(\n"
          f"      static_cast<int>(idx),\n"
          f"      svc->getRawWrite(\"data\"));")
    elif isinstance(first_port, _MMIORegionPort):
      find_parts.append(
          f"  {map_var}.try_emplace(\n"
          f"      static_cast<int>(idx),\n"
          f"      esi::findPortAsOrThrow<esi::services::MMIO::MMIORegion>(\n"
          f"          rawModule, {appid_expr}));")
    elif isinstance(first_port, _MetricPort):
      find_parts.append(
          f"  {map_var}.try_emplace(\n"
          f"      static_cast<int>(idx),\n"
          f"      esi::findPortAsOrThrow<esi::services::TelemetryService::"
          f"Metric>(\n"
          f"          rawModule, {appid_expr}));")
    else:
      # Plain BundlePort: any service port that doesn't match a standard
      # specialization (e.g. a custom `@esi.ServiceDecl`-defined service).
      find_parts.append(
          f"  {map_var}.try_emplace(\n"
          f"      static_cast<int>(idx),\n"
          f"      &esi::findPortOrThrow(rawModule, {appid_expr}));")
    find_parts.append("}")
    find_parts.append(f"{indexed_type} {indexed_var}(std::move({map_var}));")

    post = ""
    if self._port_is_connectable(first_port):
      # IndexedPorts now exposes mutable iteration so `port.connect()` is fine.
      post = (f"for (auto &[idx, port] : connected->{member_name})\n"
              f"  port.connect();")

    return _PortGroup(
        member_decl=f"{indexed_type} {member_name};",
        ctor_params=[f"{indexed_type} {indexed_var}"],
        init_entry=f"{member_name}(std::move({indexed_var}))",
        find_code="\n".join(find_parts),
        make_unique_args=[f"std::move({indexed_var})"],
        post_connect=post,
        using_aliases=aliases,
    )

  def _mixed_struct_group(self, member_name: str, appid_name: str,
                          port_list) -> _PortGroup:
    """Build a _PortGroup for a same-name, mixed-type indexed port group."""
    struct_name = f"{member_name}_ports"
    sub_member_decls: List[str] = []
    ctor_params: List[str] = []
    init_args: List[str] = []
    find_parts: List[str] = []
    make_args: List[str] = []
    post_parts: List[str] = []
    using_aliases: List[Tuple[str, str]] = []

    for appid, port in port_list:
      idx = appid.idx if appid.idx is not None else 0
      sub_name = f"_{idx}"
      sub_alias_prefix = f"{member_name}_{idx}"
      sub_aliases = self._port_using_aliases(sub_alias_prefix, port)
      using_aliases.extend(sub_aliases)
      alias_prefix = sub_alias_prefix if sub_aliases else None
      member_type = self._cpp_member_type(port, alias_prefix=alias_prefix)
      is_ref = member_type.endswith(" &")
      if is_ref:
        sub_member_decls.append(f"{member_type}{sub_name};")
      else:
        sub_member_decls.append(f"{member_type} {sub_name};")

      param_type = self._cpp_ctor_param_type(port)
      param_suffix = self._cpp_ctor_param_suffix(port)
      param_name = f"{member_name}_{idx}{param_suffix}"
      ctor_params.append(f"{param_type} {param_name}")
      init_args.append(param_name)
      find_parts.append(self._port_find_code(param_name, port, appid))
      make_args.append(self._port_make_unique_arg(param_name, port))
      if self._port_is_connectable(port):
        post_parts.append(f"connected->{member_name}.{sub_name}.connect();")

    struct_decl = (f"struct {struct_name} {{\n" +
                   "".join(f"      {d}\n" for d in sub_member_decls) + "    };")
    init_entry = f"{member_name}{{{', '.join(init_args)}}}"

    return _PortGroup(
        struct_decls=[struct_decl],
        member_decl=f"{struct_name} {member_name};",
        ctor_params=ctor_params,
        init_entry=init_entry,
        find_code="\n".join(find_parts),
        make_unique_args=make_args,
        post_connect="\n".join(post_parts),
        using_aliases=using_aliases,
    )

  def _collect_port_groups(self, ports: dict) -> List[_PortGroup]:
    """Group `ports` (AppID → BundlePort) into _PortGroup list."""
    # Group by AppID name, preserving sorted order.
    groups_by_name: Dict[str, list] = {}
    for appid, port in ports.items():
      n = appid.name
      if n not in groups_by_name:
        groups_by_name[n] = []
      groups_by_name[n].append((appid, port))

    result: List[_PortGroup] = []
    for appid_name, port_list in groups_by_name.items():
      member_name = self._sanitize_id(appid_name)
      # Sort by idx (None → -1 so scalar ports sort first).
      port_list.sort(key=lambda x: x[0].idx if x[0].idx is not None else -1)

      if len(port_list) == 1:
        appid, port = port_list[0]
        if appid.idx is None:
          result.append(self._scalar_port_group(member_name, port, appid))
        else:
          result.append(
              self._indexed_ports_group(member_name, appid_name, port_list))
        continue

      # Multiple ports with the same name.
      all_indexed = all(a.idx is not None for a, _ in port_list)
      if not all_indexed:
        # Degenerate: mix of indexed and non-indexed with the same name.
        # Emit as a mixed struct for safety.
        result.append(
            self._mixed_struct_group(member_name, appid_name, port_list))
        continue

      all_same_type = len({type(p) for _, p in port_list}) == 1
      if all_same_type:
        result.append(
            self._indexed_ports_group(member_name, appid_name, port_list))
      else:
        result.append(
            self._mixed_struct_group(member_name, appid_name, port_list))

    return result

  def _emit_module_class(self, name: str, system_name: str,
                         module_info: ModuleInfo, port_groups: List[_PortGroup],
                         out: TextIO) -> None:
    """Emit the full module class to `out`."""
    out.write(f"/// Generated header for {system_name} module {name}.\n"
              "#pragma once\n"
              '#include "types.h"\n'
              '#include "esi/TypedPorts.h"\n'
              "\n"
              "#include <any>\n"
              "#include <map>\n"
              "#include <optional>\n"
              "#include <string>\n"
              "\n"
              f"namespace {system_name} {{\n"
              "\n")

    # Module metadata as a Doxygen comment block above the class.
    metadata_lines: List[str] = []
    summary = getattr(module_info, "summary", None)
    if summary:
      for line in summary.splitlines():
        metadata_lines.append(line)
    for label, attr in (("Version", "version"), ("Repository", "repo"),
                        ("Commit", "commit_hash")):
      val = getattr(module_info, attr, None)
      if val:
        metadata_lines.append(f"{label}: {val}")
    if metadata_lines:
      out.write("///\n")
      for line in metadata_lines:
        out.write(f"/// {line}\n" if line else "///\n")
      out.write("///\n")

    out.write(f"class {name} {{\n"
              "public:\n")

    consts = self.get_consts_str(module_info)
    if consts:
      out.write("  // Module constants.\n")
      out.write(f"  {consts}\n\n")

    # Type aliases for typed-port template parameters, hoisted to module scope
    # so the long mangled names don't appear inline as template arguments.
    aliases = [a for grp in port_groups for a in grp.using_aliases]
    if aliases:
      for alias_name, alias_type in aliases:
        out.write(f"  using {alias_name} = {alias_type};\n")
      out.write("\n")

    if port_groups:
      out.write(
          "  /// Holds the resolved, typed ports for this module instance.\n"
          "  /// Returned by `connect()`.\n"
          "  class Connected {\n  public:\n")

      # Struct declarations for mixed groups.
      for grp in port_groups:
        for decl in grp.struct_decls:
          out.write(f"    {decl}\n")
      if any(grp.struct_decls for grp in port_groups):
        out.write("\n")

      # Member declarations.
      for grp in port_groups:
        out.write(f"    {grp.member_decl}\n")
      out.write("\n")

      # Constructor.
      all_params = [p for grp in port_groups for p in grp.ctor_params]
      out.write("    Connected(\n")
      for i, param in enumerate(all_params):
        comma = "," if i < len(all_params) - 1 else ""
        out.write(f"        {param}{comma}\n")
      out.write("        )\n        : ")
      inits = [grp.init_entry for grp in port_groups]
      out.write(",\n          ".join(inits))
      out.write(" {}\n  };\n\n")

    # Outer constructor.
    out.write(
        f"  {name}(esi::HWModule *rawModule) : rawModule(rawModule) {{}}\n\n")

    # Module-metadata accessors. These read from the live HWModule's ModuleInfo
    # so callers can verify that the connected accelerator is compatible with
    # the build the software was generated against.
    out.write(
        "  /// The connected module's name as reported by the manifest, or\n"
        "  /// std::nullopt if the module has no metadata.\n"
        "  std::optional<std::string> name() const {\n"
        "    auto info = rawModule->getInfo();\n"
        "    return info ? info->name : std::nullopt;\n"
        "  }\n"
        "  /// The connected module's summary string, if any.\n"
        "  std::optional<std::string> summary() const {\n"
        "    auto info = rawModule->getInfo();\n"
        "    return info ? info->summary : std::nullopt;\n"
        "  }\n"
        "  /// The connected module's version string, if any.\n"
        "  std::optional<std::string> version() const {\n"
        "    auto info = rawModule->getInfo();\n"
        "    return info ? info->version : std::nullopt;\n"
        "  }\n"
        "  /// The connected module's source repository, if any.\n"
        "  std::optional<std::string> repo() const {\n"
        "    auto info = rawModule->getInfo();\n"
        "    return info ? info->repo : std::nullopt;\n"
        "  }\n"
        "  /// The connected module's source commit hash, if any.\n"
        "  std::optional<std::string> commitHash() const {\n"
        "    auto info = rawModule->getInfo();\n"
        "    return info ? info->commitHash : std::nullopt;\n"
        "  }\n"
        "  /// Designer-specified constants for the connected module.\n"
        "  /// Returns an empty map if the module has no metadata.\n"
        "  std::map<std::string, esi::Constant> constants() const {\n"
        "    auto info = rawModule->getInfo();\n"
        "    return info ? info->constants\n"
        "                : std::map<std::string, esi::Constant>{};\n"
        "  }\n"
        "  /// Free-form designer-supplied metadata for the connected module.\n"
        "  /// Returns an empty map if the module has no metadata.\n"
        "  std::map<std::string, std::any> extra() const {\n"
        "    auto info = rawModule->getInfo();\n"
        "    return info ? info->extra : std::map<std::string, std::any>{};\n"
        "  }\n\n")

    if port_groups:
      out.write("  std::unique_ptr<Connected> connect() {\n")

      # Find / resolve phase.
      for grp in port_groups:
        if grp.find_code:
          for line in grp.find_code.splitlines():
            out.write(f"    {line}\n")
          out.write("\n")

      # Construct Connected.
      all_args = [a for grp in port_groups for a in grp.make_unique_args]
      out.write("    auto connected = std::make_unique<Connected>(\n")
      for i, arg in enumerate(all_args):
        comma = "," if i < len(all_args) - 1 else ""
        out.write(f"        {arg}{comma}\n")
      out.write("    );\n\n")

      # Post-construction connects.
      for grp in port_groups:
        if grp.post_connect:
          for line in grp.post_connect.splitlines():
            out.write(f"    {line}\n")

      out.write("    return connected;\n  }\n\n")

    out.write("private:\n  esi::HWModule *rawModule;\n};\n\n")
    out.write(f"}} // namespace {system_name}\n")

  def write_modules(self, output_dir: Path, system_name: str):
    """Write the C++ header. One for each module in the manifest."""
    module_instances = self._build_module_instance_map()

    for module_info in self.manifest.module_infos:
      if module_info.name is None:
        continue
      name = module_info.name
      instance = module_instances.get(name)
      if instance is not None:
        port_groups = self._collect_port_groups(instance.ports)
      else:
        port_groups = []

      hdr_file = output_dir / f"{name}.h"
      with open(hdr_file, "w") as hdr:
        self._emit_module_class(name, system_name, module_info, port_groups,
                                hdr)

  def generate(self, output_dir: Path, system_name: str):
    self.type_emitter.write_header(output_dir, system_name)
    self.write_modules(output_dir, system_name)


class CppTypePlanner:
  """Plan C++ type naming and ordering from an ESI manifest."""

  def __init__(self, type_table) -> None:
    """Initialize the generator with the manifest and target namespace."""
    # Map manifest type ids to their preferred C++ names.
    self.type_id_map: Dict[types.ESIType, str] = {}
    # Track all names already taken to avoid collisions. True => alias-based.
    self.used_names: Dict[str, bool] = {}
    # Track alias base names to warn on collisions.
    self.alias_base_names: Set[str] = set()
    self.ordered_types: List[types.ESIType] = []
    self.has_cycle = False
    self._prepare_types(type_table)

  def _prepare_types(self, type_table) -> None:
    """Name the types and prepare for emission by registering all reachable
    types and assigning."""
    visited: Set[str] = set()
    for t in type_table:
      self._collect_aliases(t, visited)

    visited = set()
    for t in type_table:
      self._collect_structs(t, visited)

    visited = set()
    for t in type_table:
      self._collect_windows(t, visited)

    self.ordered_types, self.has_cycle = self._ordered_emit_types()

  def _sanitize_name(self, name: str) -> str:
    """Create a C++-safe identifier from the manifest-provided name."""
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

  def _reserve_name(self, base: str, is_alias: bool) -> str:
    """Reserve a globally unique identifier using the sanitized base name."""
    base = self._sanitize_name(base)
    if is_alias and base in self.alias_base_names:
      sys.stderr.write(
          f"Warning: duplicate alias name '{base}' detected; disambiguating.\n")
    if is_alias:
      self.alias_base_names.add(base)
    name = base
    idx = 1
    while name in self.used_names:
      name = f"{base}_{idx}"
      idx += 1
    self.used_names[name] = is_alias
    return name

  def _auto_struct_name(self, struct_type: types.StructType) -> str:
    """Derive a deterministic name for anonymous structs from their fields."""
    parts = ["_struct"]
    for field_name, field_type in struct_type.fields:
      parts.append(field_name)
      parts.append(self._sanitize_name(field_type.id))
    return self._reserve_name("_".join(parts), is_alias=False)

  def _auto_union_name(self, union_type: types.UnionType) -> str:
    """Derive a deterministic name for anonymous unions from their fields."""
    parts = ["_union"]
    for field_name, field_type in union_type.fields:
      parts.append(field_name)
      parts.append(self._sanitize_name(field_type.id))
    return self._reserve_name("_".join(parts), is_alias=False)

  def _auto_window_name(self, window_type: types.WindowType) -> str:
    """Derive a deterministic name for generated window helpers.

    Two distinct windows can wrap the same `into` struct (e.g. serial and
    parallel encodings of the same payload), so the helper name must be
    derived from BOTH the inner type's name and the window's own name/id.
    """
    into_type = self._unwrap_aliases(window_type.into_type)
    into_name = self.type_id_map.get(into_type)
    window_part = (window_type.name
                   if window_type.name else self._sanitize_name(window_type.id))
    if into_name:
      base = f"{into_name}_{window_part}"
    elif window_type.name:
      base = window_part
    else:
      base = f"_window_{window_part}"
    return self._reserve_name(base, is_alias=False)

  def _unwrap_aliases(self, wrapped: types.ESIType) -> types.ESIType:
    while isinstance(wrapped, types.TypeAlias):
      wrapped = wrapped.inner_type
    return wrapped

  def _is_supported_window(self, current_type: types.ESIType) -> bool:
    if not isinstance(current_type, types.WindowType):
      return False
    into_type = self._unwrap_aliases(current_type.into_type)
    if not isinstance(into_type, types.StructType):
      return False

    # The generated window helper only supports struct-shaped payloads with a
    # single logical list field to stream across multiple frames.
    list_fields = []
    for field_name, field_type in into_type.fields:
      if isinstance(self._unwrap_aliases(field_type), types.ListType):
        list_fields.append(field_name)
    if len(list_fields) != 1:
      return False

    list_field_name = list_fields[0]
    header_field = None
    data_field = None
    # That list must appear exactly once as a bulk-count field and exactly once
    # as a single-item data field so the helper can synthesize header/data/footer.
    for frame in current_type.frames:
      for field in frame.fields:
        if field.name != list_field_name:
          continue
        if field.bulk_count_width > 0:
          if header_field is not None:
            return False
          header_field = field
        elif field.num_items > 0:
          if data_field is not None:
            return False
          data_field = field
    return (header_field is not None and data_field is not None and
            data_field.num_items == 1)

  def _iter_type_children(self, t: types.ESIType) -> List[types.ESIType]:
    """Return child types in a stable order for traversal."""
    if isinstance(t, types.TypeAlias):
      return [t.inner_type] if t.inner_type is not None else []
    if isinstance(t, types.BundleType):
      return [channel.type for channel in t.channels]
    if isinstance(t, types.ChannelType):
      return [t.inner]
    if isinstance(t, types.StructType):
      return [field_type for _, field_type in t.fields]
    if isinstance(t, types.UnionType):
      return [field_type for _, field_type in t.fields]
    if isinstance(t, types.ListType):
      return [t.element_type]
    if isinstance(t, types.WindowType):
      return [t.into_type]
    if isinstance(t, types.ArrayType):
      return [t.element_type]
    return []

  def _visit_types(self, t: types.ESIType, visited: Set[str], visit_fn) -> None:
    """Traverse types with alphabetical child ordering in post-order."""
    if not isinstance(t, types.ESIType):
      raise TypeError(f"Expected ESIType, got {type(t)}")
    tid = t.id
    if tid in visited:
      return
    visited.add(tid)
    children = sorted(self._iter_type_children(t), key=lambda child: child.id)
    for child in children:
      self._visit_types(child, visited, visit_fn)
    visit_fn(t)

  def _collect_aliases(self, t: types.ESIType, visited: Set[str]) -> None:
    """Scan for aliases and reserve their names (recursive)."""

    # Visit callback: reserve alias names and map aliases to identifiers.
    def visit(alias_type: types.ESIType) -> None:
      if not isinstance(alias_type, types.TypeAlias):
        return
      if alias_type not in self.type_id_map:
        alias_name = self._reserve_name(alias_type.name, is_alias=True)
        self.type_id_map[alias_type] = alias_name

    self._visit_types(t, visited, visit)

  def _collect_structs(self, t: types.ESIType, visited: Set[str]) -> None:
    """Scan for structs/unions needing auto-names and reserve them."""

    # Visit callback: assign auto-names to unnamed structs and unions.
    def visit(current_type: types.ESIType) -> None:
      if current_type in self.type_id_map:
        return
      if isinstance(current_type, types.StructType):
        self.type_id_map[current_type] = self._auto_struct_name(current_type)
      elif isinstance(current_type, types.UnionType):
        self.type_id_map[current_type] = self._auto_union_name(current_type)

    self._visit_types(t, visited, visit)

  def _collect_windows(self, t: types.ESIType, visited: Set[str]) -> None:
    """Scan for supported window types and reserve helper names."""

    def visit(current_type: types.ESIType) -> None:
      if not self._is_supported_window(current_type):
        return
      assert isinstance(current_type, types.WindowType)
      if current_type in self.type_id_map:
        return
      self.type_id_map[current_type] = self._auto_window_name(current_type)

    self._visit_types(t, visited, visit)

  def _collect_decls_from_type(self,
                               wrapped: types.ESIType) -> Set[types.ESIType]:
    """Collect types that require top-level declarations for a given type."""
    deps: Set[types.ESIType] = set()

    # Visit callback: collect structs, unions, and non-struct aliases used by a
    # type.
    def visit(current: types.ESIType) -> None:
      if isinstance(current, types.TypeAlias):
        inner = current.inner_type
        if inner is not None and (isinstance(
            inner, (types.StructType, types.UnionType)) or
                                  self._is_supported_window(inner)):
          deps.add(inner)
        else:
          deps.add(current)
      elif isinstance(current, (types.StructType, types.UnionType)):
        deps.add(current)
      elif self._is_supported_window(current):
        deps.add(current)

    self._visit_types(wrapped, set(), visit)
    return deps

  def _collect_decls_from_window(
      self, window_type: types.WindowType) -> Set[types.ESIType]:
    """Collect only the declarations referenced by a generated window helper."""
    deps: Set[types.ESIType] = set()
    into_type = self._unwrap_aliases(window_type.into_type)
    if not isinstance(into_type, types.StructType):
      return deps

    for _, field_type in into_type.fields:
      unwrapped = self._unwrap_aliases(field_type)
      if isinstance(unwrapped, types.ListType):
        deps.update(self._collect_decls_from_type(unwrapped.element_type))
      else:
        deps.update(self._collect_decls_from_type(field_type))
    return deps

  def _ordered_emit_types(self) -> Tuple[List[types.ESIType], bool]:
    """Collect and order types for deterministic emission."""
    window_into_types: Set[types.ESIType] = set()
    for esi_type in self.type_id_map.keys():
      if not self._is_supported_window(esi_type):
        continue
      assert isinstance(esi_type, types.WindowType)
      window_into_types.add(self._unwrap_aliases(esi_type.into_type))
    emit_types: List[types.ESIType] = []
    for esi_type in self.type_id_map.keys():
      if isinstance(esi_type,
                    types.StructType) and esi_type in window_into_types:
        continue
      if isinstance(esi_type, types.TypeAlias):
        inner = esi_type.inner_type
        if inner is not None and self._unwrap_aliases(
            inner) in window_into_types:
          continue
      if (isinstance(esi_type,
                     (types.StructType, types.UnionType, types.TypeAlias)) or
          self._is_supported_window(esi_type)):
        emit_types.append(esi_type)

    # Prefer alias-reserved names first, then lexicographic for determinism.
    name_to_type = {self.type_id_map[t]: t for t in emit_types}
    sorted_names = sorted(name_to_type.keys(),
                          key=lambda name:
                          (0 if self.used_names.get(name, False) else 1, name))

    ordered: List[types.ESIType] = []
    visited: Set[types.ESIType] = set()
    visiting: Set[types.ESIType] = set()
    has_cycle = False

    # Visit callback: DFS to emit dependencies before their users.
    def visit(current: types.ESIType) -> None:
      nonlocal has_cycle
      if current in visited:
        return
      if current in visiting:
        has_cycle = True
        return
      visiting.add(current)

      deps: Set[types.ESIType] = set()
      if isinstance(current, types.TypeAlias):
        inner = current.inner_type
        if inner is not None:
          deps.update(self._collect_decls_from_type(inner))
      elif isinstance(current, (types.StructType, types.UnionType)):
        for _, field_type in current.fields:
          deps.update(self._collect_decls_from_type(field_type))
      elif self._is_supported_window(current):
        assert isinstance(current, types.WindowType)
        deps.update(self._collect_decls_from_window(current))
      for dep in sorted(deps, key=lambda dep: self.type_id_map[dep]):
        visit(dep)

      visiting.remove(current)
      visited.add(current)
      ordered.append(current)

    for name in sorted_names:
      visit(name_to_type[name])

    return ordered, has_cycle


class CppTypeEmitter:
  """Emit C++ headers from precomputed type ordering."""

  def __init__(self, planner: CppTypePlanner) -> None:
    self.type_id_map = planner.type_id_map
    self.ordered_types = planner.ordered_types
    self.has_cycle = planner.has_cycle

  def type_identifier(self, type: types.ESIType) -> str:
    """Get the C++ type string for an ESI type."""
    return self._cpp_type(type)

  def _cpp_string_literal(self, value: str) -> str:
    """Escape a Python string for use as a C++ string literal."""
    escaped = value.replace("\\", "\\\\").replace('"', '\\"')
    return f'"{escaped}"'

  def _get_bitvector_str(self, type: types.ESIType) -> str:
    """Get the textual code for the storage class of an integer type."""
    assert isinstance(type, (types.BitsType, types.IntType))

    return self._storage_type(
        type.bit_width, not isinstance(type, (types.BitsType, types.UIntType)))

  def _storage_type(self, bit_width: int, signed: bool) -> str:
    """Get the textual code for a byte-addressable integer storage type."""

    if bit_width == 1:
      return "bool"
    elif bit_width <= 8:
      storage_width = 8
    elif bit_width <= 16:
      storage_width = 16
    elif bit_width <= 32:
      storage_width = 32
    elif bit_width <= 64:
      storage_width = 64
    else:
      raise ValueError(f"Unsupported integer width: {bit_width}")

    if not signed:
      return f"uint{storage_width}_t"
    return f"int{storage_width}_t"

  def _type_byte_width(self, wrapped: types.ESIType) -> int:
    """Return the size of a fixed-width type in bytes."""
    if wrapped.bit_width < 0:
      raise ValueError(f"Unsupported unbounded type width for '{wrapped}'")
    return (wrapped.bit_width + 7) // 8

  def _array_base_and_dims(
      self, array_type: types.ArrayType) -> Tuple[str, List[int]]:
    """Return the base C++ type and outer-to-inner dimensions of a nested array."""
    dims: List[int] = []
    inner: types.ESIType = array_type
    while isinstance(inner, types.ArrayType):
      dims.append(inner.size)
      inner = inner.element_type
    base_cpp = self._cpp_type(inner)
    return base_cpp, dims

  def _std_array_type(self, array_type: types.ArrayType) -> str:
    """Return the equivalent nested `std::array<...>` type for an array.

    `std::array<T, N>` is layout-compatible in practice with `T[N]` on every
    major implementation (and identical under `#pragma pack(1)`), so the
    generator uses it everywhere a fixed-size array would appear.  This keeps
    field/value/ctor types storable in `std::vector` and assignable with `=`.
    """
    base_cpp, dims = self._array_base_and_dims(array_type)
    result = base_cpp
    for size in reversed(dims):
      result = f"std::array<{result}, {size}>"
    return result

  def _cpp_type(self, wrapped: types.ESIType) -> str:
    """Resolve an ESI type to its C++ identifier."""
    if isinstance(wrapped, types.WindowType) and wrapped in self.type_id_map:
      return self.type_id_map[wrapped]
    if isinstance(wrapped,
                  (types.TypeAlias, types.StructType, types.UnionType)):
      return self.type_id_map[wrapped]
    if isinstance(wrapped, types.BundleType):
      return "void"
    if isinstance(wrapped, types.ChannelType):
      return self._cpp_type(wrapped.inner)
    if isinstance(wrapped, types.ListType):
      raise ValueError("List types require a generated window wrapper")
    if isinstance(wrapped, types.VoidType):
      return "void"
    if isinstance(wrapped, types.AnyType):
      return "std::any"
    if isinstance(wrapped, (types.BitsType, types.IntType)):
      return self._get_bitvector_str(wrapped)
    if isinstance(wrapped, types.ArrayType):
      return self._std_array_type(wrapped)
    if type(wrapped) is types.ESIType:
      return "std::any"
    raise NotImplementedError(
        f"Type '{wrapped}' not supported for C++ generation")

  def _unwrap_aliases(self, wrapped: types.ESIType) -> types.ESIType:
    """Strip alias wrappers to reach the underlying type."""
    while isinstance(wrapped, types.TypeAlias):
      wrapped = wrapped.inner_type
    return wrapped

  def _format_window_field_decl(self, field_name: str,
                                field_type: types.ESIType) -> str:
    """Emit a packed field declaration for generated window helpers.

    Arrays use `std::array` (handled by `_cpp_type`), so no bracket syntax is
    needed and a single uniform declaration form covers every type.
    """
    field_cpp = self._cpp_type(field_type)
    wrapped = self._unwrap_aliases(field_type)
    if isinstance(wrapped, (types.BitsType, types.IntType)) and \
       wrapped.bit_width % 8 != 0:
      return f"{field_cpp} {field_name} : {wrapped.bit_width};"
    return f"{field_cpp} {field_name};"

  def _format_window_ctor_param(self, field_name: str,
                                field_type: types.ESIType) -> str:
    """Emit a constructor parameter for generated window helpers.

    Small scalar header fields are cheaper to pass by value than by reference.
    Larger aggregates stay as const references.
    """
    field_cpp = self._cpp_type(field_type)
    wrapped = self._unwrap_aliases(field_type)
    if isinstance(wrapped, (types.BitsType, types.IntType)):
      return f"{field_cpp} {field_name}"
    return f"const {field_cpp} &{field_name}"

  def _emit_window_field_copy(self, hdr: TextIO, dest_expr: str, src_expr: str,
                              field_type: types.ESIType) -> None:
    """Copy a generated window field."""
    hdr.write(f"    {dest_expr} = {src_expr};\n")

  def _field_byte_width(self, field_type: types.ESIType) -> int:
    """Compute the byte width of a field type, rounding up to full bytes."""
    return (field_type.bit_width + 7) // 8

  def _safe_byte_width(self, esi_type: types.ESIType) -> Optional[int]:
    """Return the bounded byte width of `esi_type`, or `None` if it has no
    well-defined static size (e.g. unbounded `!esi.any` or recursive types).
    """
    try:
      bit_width = esi_type.bit_width
    except Exception:
      return None
    if bit_width is None or bit_width < 0:
      return None
    return (bit_width + 7) // 8

  def _emit_size_assert(self,
                        hdr: TextIO,
                        type_name: str,
                        expected_bytes: Optional[int],
                        indent: str = "") -> None:
    """Emit a `static_assert` that pins the C++ `sizeof` of a packed type to
    the byte width derived from the manifest.

    `std::array` and bit-field layout are technically implementation-defined,
    so this assertion is the safety net that catches a toolchain that lays
    them out differently from the wire format.  Skipped silently for types
    without a bounded static size.
    """
    if expected_bytes is None:
      return
    hdr.write(
        f"{indent}static_assert(sizeof({type_name}) == {expected_bytes},\n"
        f"{indent}              \"{type_name}: packed layout does not match "
        f"manifest size\");\n")

  def _analyze_window(self, window_type: types.WindowType):
    """Extract the metadata needed to emit a bulk list window wrapper."""
    into_type = self._unwrap_aliases(window_type.into_type)
    if not isinstance(into_type, types.StructType):
      raise ValueError("window codegen currently requires a struct into-type")

    field_map = {name: field_type for name, field_type in into_type.fields}
    list_fields = [
        (name, self._unwrap_aliases(field_type))
        for name, field_type in into_type.fields
        if isinstance(self._unwrap_aliases(field_type), types.ListType)
    ]
    if len(list_fields) != 1:
      raise ValueError("window codegen currently supports exactly one list")

    list_field_name, list_type = list_fields[0]
    assert isinstance(list_type, types.ListType)

    header_frame = None
    header_field = None
    data_frame = None
    data_field = None
    for frame in window_type.frames:
      for field in frame.fields:
        if field.name != list_field_name:
          continue
        if field.bulk_count_width > 0:
          header_frame = frame
          header_field = field
        elif field.num_items > 0:
          data_frame = frame
          data_field = field

    if header_frame is None or header_field is None:
      raise ValueError("window codegen requires a bulk-count header frame")
    if data_frame is None or data_field is None:
      raise ValueError("window codegen requires a data frame for the list")
    if data_field.num_items != 1:
      raise ValueError("window codegen currently supports numItems == 1")

    ctor_params = [(name, field_type)
                   for name, field_type in into_type.fields
                   if name != list_field_name]

    header_fields = []
    header_bytes = 0
    count_field_name = f"{list_field_name}_count"
    count_width = header_field.bulk_count_width
    count_cpp = self._storage_type(count_width, signed=False)
    count_bytes = (count_width + 7) // 8
    for field in reversed(header_frame.fields):
      if field.name == list_field_name:
        header_fields.append((count_field_name, None))
        header_bytes += count_bytes
      else:
        field_type = field_map[field.name]
        header_fields.append((field.name, field_type))
        header_bytes += self._type_byte_width(field_type)

    data_fields = []
    data_bytes = 0
    for field in reversed(data_frame.fields):
      if field.name == list_field_name:
        data_fields.append((list_field_name, list_type.element_type))
        data_bytes += self._type_byte_width(list_type.element_type)
      else:
        field_type = field_map[field.name]
        data_fields.append((field.name, field_type))
        data_bytes += self._type_byte_width(field_type)

    frame_bytes = max(header_bytes, data_bytes)

    return {
        "ctor_params": ctor_params,
        "count_cpp": count_cpp,
        "count_field_name": count_field_name,
        "count_width": count_width,
        "data_fields": data_fields,
        "data_pad_bytes": frame_bytes - data_bytes,
        "element_cpp": self._cpp_type(list_type.element_type),
        "frame_bytes": frame_bytes,
        "header_fields": header_fields,
        "header_pad_bytes": frame_bytes - header_bytes,
        "list_field_name": list_field_name,
        "window_name": self.type_id_map[window_type],
    }

  def _emit_struct(self, hdr: TextIO, struct_type: types.StructType) -> None:
    """Emit a packed struct declaration plus its type id string."""
    fields = list(struct_type.fields)
    if struct_type.cpp_type.reverse:
      fields = list(reversed(fields))
    field_decls: List[str] = []
    for field_name, field_type in fields:
      field_cpp = self._cpp_type(field_type)
      wrapped = self._unwrap_aliases(field_type)
      if isinstance(wrapped, (types.BitsType, types.IntType)):
        # TODO: Bitfield layout is implementation-defined; consider
        # byte-aligned storage with explicit pack/unpack helpers.
        bitfield_width = wrapped.bit_width
        if bitfield_width >= 0:
          field_decls.append(f"{field_cpp} {field_name} : {bitfield_width};")
        else:
          field_decls.append(f"{field_cpp} {field_name};")
      else:
        field_decls.append(f"{field_cpp} {field_name};")
    struct_name = self.type_id_map[struct_type]
    hdr.write("#pragma pack(push, 1)\n")
    hdr.write(f"struct {struct_name} {{\n")
    for decl in field_decls:
      hdr.write(f"  {decl}\n")
    hdr.write("\n")
    # Logical-order constructor: parameters follow `struct_type.fields` order
    # (the user-facing order from the manifest), independent of any wire-layout
    # reversal applied to the member declarations above.
    logical_fields = list(struct_type.fields)
    if logical_fields:
      hdr.write(f"  {struct_name}() = default;\n")
      ctor_params = ", ".join(
          self._format_window_ctor_param(name, ftype)
          for name, ftype in logical_fields)
      inits = ", ".join(f"{name}({name})" for name, _ in logical_fields)
      hdr.write(f"  {struct_name}({ctor_params}) : {inits} {{}}\n\n")
    hdr.write(
        f"  static constexpr std::string_view _ESI_ID = {self._cpp_string_literal(struct_type.id)};\n"
    )
    hdr.write("};\n")
    self._emit_size_assert(hdr, struct_name, self._safe_byte_width(struct_type))
    hdr.write("#pragma pack(pop)\n\n")

  def _emit_union(self, hdr: TextIO, union_type: types.UnionType) -> None:
    """Emit a packed union declaration plus its type id string.

    Fields narrower than the union width get wrapper structs with a `_pad`
    byte array so the data sits at the MSB end, matching SV packed union
    layout where padding occupies the LSBs / lower addresses.
    """
    union_name = self.type_id_map[union_type]
    union_bytes = self._field_byte_width(union_type)
    fields = list(union_type.fields)

    hdr.write("#pragma pack(push, 1)\n")
    # First pass: emit wrapper structs for fields that need padding.
    wrapper_names: Dict[str, str] = {}
    for field_name, field_type in fields:
      field_bytes = self._field_byte_width(field_type)
      pad_bytes = union_bytes - field_bytes
      if pad_bytes > 0:
        wrapper = f"{union_name}_{field_name}"
        wrapper_names[field_name] = wrapper
        hdr.write(f"struct {wrapper} {{\n")
        hdr.write(f"  uint8_t _pad[{pad_bytes}];\n")
        field_cpp = self._cpp_type(field_type)
        hdr.write(f"  {field_cpp} {field_name};\n")
        hdr.write("};\n")
        self._emit_size_assert(hdr, wrapper, union_bytes)

    # Second pass: emit the union itself.
    union_field_decls: List[str] = []
    for field_name, field_type in fields:
      if field_name in wrapper_names:
        union_field_decls.append(f"{wrapper_names[field_name]} {field_name};")
      else:
        field_cpp = self._cpp_type(field_type)
        union_field_decls.append(f"{field_cpp} {field_name};")
    hdr.write(f"union {union_name} {{\n")
    for decl in union_field_decls:
      hdr.write(f"  {decl}\n")
    hdr.write("\n")
    hdr.write(
        f"  static constexpr std::string_view _ESI_ID = {self._cpp_string_literal(union_type.id)};\n"
    )
    hdr.write("};\n")
    self._emit_size_assert(hdr, union_name, union_bytes)
    hdr.write("#pragma pack(pop)\n\n")

  def _emit_window(self, hdr: TextIO, window_type: types.WindowType) -> None:
    """Emit a SegmentedMessageData helper for a serial list window."""
    info = self._analyze_window(window_type)
    ctor_params = [
        self._format_window_ctor_param(name, field_type)
        for name, field_type in info["ctor_params"]
    ]
    value_ctor_params = list(ctor_params)
    value_ctor_params.append(
        f"const std::vector<value_type> &{info['list_field_name']}")
    value_ctor_signature = ", ".join(value_ctor_params)
    frame_ctor_params = list(ctor_params)
    frame_ctor_params.append("std::vector<data_frame> frames")
    frame_ctor_signature = ", ".join(frame_ctor_params)
    helper_args = ", ".join(name for name, _ in info["ctor_params"])
    helper_call = f"{helper_args}, std::move(frames)" if helper_args else "std::move(frames)"

    hdr.write(
        f"struct {info['window_name']} : public esi::SegmentedMessageData {{\n")
    hdr.write("public:\n")
    hdr.write(f"  using value_type = {info['element_cpp']};\n")
    hdr.write(f"  using count_type = {info['count_cpp']};\n\n")
    hdr.write("#pragma pack(push, 1)\n")
    hdr.write("  struct data_frame {\n")
    if info["data_pad_bytes"] > 0:
      hdr.write(f"    uint8_t _pad[{info['data_pad_bytes']}];\n")
    for field_name, field_type in info["data_fields"]:
      decl = self._format_window_field_decl(field_name, field_type)
      hdr.write(f"    {decl}\n")
    hdr.write("  };\n")
    self._emit_size_assert(hdr, "data_frame", info["frame_bytes"], indent="  ")
    hdr.write("#pragma pack(pop)\n\n")
    hdr.write("private:\n")
    hdr.write("#pragma pack(push, 1)\n")
    hdr.write("  struct header_frame {\n")
    if info["header_pad_bytes"] > 0:
      hdr.write(f"    uint8_t _pad[{info['header_pad_bytes']}];\n")
    for field_name, field_type in info["header_fields"]:
      if field_type is None:
        if info["count_width"] % 8 == 0:
          decl = f"count_type {field_name};"
        else:
          decl = f"count_type {field_name} : {info['count_width']};"
      else:
        decl = self._format_window_field_decl(field_name, field_type)
      hdr.write(f"    {decl}\n")
    hdr.write("  };\n")
    self._emit_size_assert(hdr,
                           "header_frame",
                           info["frame_bytes"],
                           indent="  ")
    hdr.write("#pragma pack(pop)\n\n")
    hdr.write("  header_frame header{};\n")
    hdr.write("  std::vector<data_frame> data_frames;\n")
    hdr.write("  header_frame footer{};\n\n")
    hdr.write(f"  void construct({frame_ctor_signature}) {{\n")
    hdr.write("    if (frames.empty())\n")
    hdr.write(
        f"      throw std::invalid_argument(\"{info['window_name']}: bulk windowed lists cannot be empty\");\n"
    )
    hdr.write(
        "    if (frames.size() > std::numeric_limits<count_type>::max())\n")
    hdr.write(
        f"      throw std::invalid_argument(\"{info['window_name']}: list too large for encoded count\");\n"
    )
    hdr.write(
        f"    header.{info['count_field_name']} = static_cast<count_type>(frames.size());\n"
    )
    for name, _ in info["ctor_params"]:
      field_type = next(
          field_type for field_name, field_type in info["ctor_params"]
          if field_name == name)
      self._emit_window_field_copy(hdr, f"header.{name}", name, field_type)
    hdr.write(f"    footer.{info['count_field_name']} = 0;\n")
    hdr.write("    data_frames = std::move(frames);\n")
    hdr.write("  }\n\n")
    hdr.write("public:\n")
    hdr.write(f"  {info['window_name']}({frame_ctor_signature}) {{\n")
    hdr.write(f"    construct({helper_call});\n")
    hdr.write("  }\n\n")
    hdr.write(f"  {info['window_name']}({value_ctor_signature}) {{\n")
    hdr.write("    std::vector<data_frame> frames;\n")
    hdr.write(f"    frames.reserve({info['list_field_name']}.size());\n")
    hdr.write(f"    for (const auto &element : {info['list_field_name']}) {{\n")
    hdr.write("      auto &frame = frames.emplace_back();\n")
    hdr.write(f"      frame.{info['list_field_name']} = element;\n")
    hdr.write("    }\n")
    hdr.write(f"    construct({helper_call});\n")
    hdr.write("  }\n\n")
    hdr.write("  size_t numSegments() const override { return 3; }\n")
    hdr.write("  esi::Segment segment(size_t idx) const override {\n")
    hdr.write("    if (idx == 0)\n")
    hdr.write(
        "      return {reinterpret_cast<const uint8_t *>(&header), sizeof(header)};\n"
    )
    hdr.write("    if (idx == 1)\n")
    hdr.write(
        "      return {reinterpret_cast<const uint8_t *>(data_frames.data()),\n"
    )
    hdr.write("              data_frames.size() * sizeof(data_frame)};\n")
    hdr.write("    if (idx == 2)\n")
    hdr.write(
        "      return {reinterpret_cast<const uint8_t *>(&footer), sizeof(footer)};\n"
    )
    hdr.write(
        f"    throw std::out_of_range(\"{info['window_name']}: invalid segment index\");\n"
    )
    hdr.write("  }\n\n")
    hdr.write(
        f"  static constexpr std::string_view _ESI_ID = {self._cpp_string_literal(self._unwrap_aliases(window_type.into_type).id)};\n"
    )
    # The into-type id alone cannot distinguish two different windows over
    # the same underlying struct (e.g. serial vs. parallel list encoding).
    # Emit the window id so the runtime can verify the wire format too.
    hdr.write(
        f"  static constexpr std::string_view _ESI_WINDOW_ID = {self._cpp_string_literal(window_type.id)};\n"
    )
    self._emit_window_data_accessors(hdr, info)
    self._emit_window_deserializer(hdr, info)
    hdr.write("};\n\n")

  def _emit_window_data_accessors(self, hdr: TextIO, info) -> None:
    """Emit accessors for the header and data fields of a window helper.

    Exposes each static header field as a scalar accessor, the count of data
    frames, and one vector-valued accessor per data field so decoded values
    are easy to inspect on the read side.
    """
    list_field_name = info["list_field_name"]
    hdr.write("\n")
    for field_name, field_type in info["header_fields"]:
      # Skip the synthetic bulk-count field; it is exposed via
      # `<list>_count()` below.
      if field_type is None:
        continue
      cpp = self._cpp_type(field_type)
      unwrapped_header = self._unwrap_aliases(field_type)
      # Aggregate types (structs/unions/std::array) get a const-ref accessor;
      # bit-vector scalars are returned by value.
      if isinstance(unwrapped_header,
                    (types.BitsType, types.IntType, types.VoidType)):
        hdr.write(
            f"  {cpp} {field_name}() const {{ return header.{field_name}; }}\n")
      else:
        hdr.write(
            f"  const {cpp} &{field_name}() const {{ return header.{field_name}; }}\n"
        )
    hdr.write(
        f"  size_t {list_field_name}_count() const {{ return data_frames.size(); }}\n"
    )
    for field_name, field_type in info["data_fields"]:
      # All field types — scalars, structs, and arrays-as-`std::array` — are
      # uniformly addressable via pointer-to-member, except bit-fields which
      # have no pointer-to-member representation.
      unwrapped_data = self._unwrap_aliases(field_type)
      if field_name == list_field_name:
        elem_cpp = "value_type"
      else:
        elem_cpp = self._cpp_type(field_type)
      # C++ does not allow forming a pointer-to-member for a bit-field, so for
      # non-byte-aligned integer fields we fall back to a lambda projection
      # (which copies by value on each dereference) instead of a
      # pointer-to-member projection.
      is_bitfield = (isinstance(unwrapped_data,
                                (types.BitsType, types.IntType)) and
                     unwrapped_data.bit_width % 8 != 0)
      if is_bitfield:
        projection = f"[](const data_frame &f) {{ return f.{field_name}; }}"
      else:
        projection = f"&data_frame::{field_name}"
      hdr.write(
          f"  auto {field_name}() const {{\n"
          f"    return std::views::transform(data_frames, {projection});\n"
          f"  }}\n")
      hdr.write(f"  std::vector<{elem_cpp}> {field_name}_vector() const {{\n"
                f"    std::vector<{elem_cpp}> out;\n"
                f"    out.reserve(data_frames.size());\n"
                f"    for (const auto &frame : data_frames)\n"
                f"      out.push_back(frame.{field_name});\n"
                f"    return out;\n"
                f"  }}\n")

  def _emit_window_deserializer(self, hdr: TextIO, info) -> None:
    """Emit a few bridge helpers + a `TypeDeserializer` alias.

    The actual decoder lives in `esi::SerialListTypeDeserializer<T>`, which
    walks the header/data/footer burst protocol generically. Each window
    helper only has to expose:

      - `_headerCount(const header_frame &)` -> `count_type`
      - `_fromFrames(const header_frame &, std::vector<data_frame> &&)`
        -> `std::unique_ptr<T>`

    plus a `friend class esi::SerialListTypeDeserializer<T>;` so the template
    can reach the (private) `header_frame` definition.
    """
    window_name = info["window_name"]
    count_field_name = info["count_field_name"]
    ctor_args = ", ".join(f"h.{name}" for name, _ in info["ctor_params"])
    if ctor_args:
      ctor_args = f"{ctor_args}, std::move(frames)"
    else:
      ctor_args = "std::move(frames)"

    hdr.write("\n")
    hdr.write("private:\n")
    hdr.write(
        "  // Bridge helpers used by esi::SerialListTypeDeserializer<T>; the\n")
    hdr.write(
        "  // template walks the serial-list burst protocol generically and\n")
    hdr.write(
        "  // reaches into `header_frame` via the friend declaration below.\n")
    hdr.write("  static count_type _headerCount(const header_frame &h) {\n")
    hdr.write(f"    return h.{count_field_name};\n")
    hdr.write("  }\n")
    hdr.write(f"  static std::unique_ptr<{window_name}> _fromFrames(\n")
    hdr.write(
        "      const header_frame &h, std::vector<data_frame> &&frames) {\n")
    hdr.write(f"    return std::make_unique<{window_name}>({ctor_args});\n")
    hdr.write("  }\n")
    hdr.write(
        f"  friend class esi::SerialListTypeDeserializer<{window_name}>;\n\n")
    hdr.write("public:\n")
    hdr.write(
        f"  using TypeDeserializer = esi::SerialListTypeDeserializer<{window_name}>;\n"
    )

  def _emit_alias(self, hdr: TextIO, alias_type: types.TypeAlias) -> None:
    """Emit a using alias when the alias targets a different C++ type."""
    inner_wrapped = alias_type.inner_type
    alias_name = self.type_id_map[alias_type]
    inner_cpp = None
    if inner_wrapped is not None:
      inner_cpp = self._cpp_type(inner_wrapped)
    if inner_cpp is None:
      inner_cpp = self.type_id_map[alias_type]
    if inner_cpp != alias_name:
      hdr.write(f"using {alias_name} = {inner_cpp};\n\n")

  def write_header(self, output_dir: Path, system_name: str) -> None:
    """Emit the fully ordered types.h header into the output directory."""
    hdr_file = output_dir / "types.h"
    with open(hdr_file, "w") as hdr:
      hdr.write(
          textwrap.dedent(f"""
        // Generated header for {system_name} types.
        #pragma once

        #include <cstdint>
        #include <cstddef>
        #include <any>
        #include <array>
        #include <limits>
        #include <ranges>
        #include <stdexcept>
        #include <string_view>
        #include <utility>
        #include <vector>

        #include "esi/Common.h"
        #include "esi/TypedPorts.h"

        namespace {system_name} {{

      """))
      if self.has_cycle:
        sys.stderr.write("Warning: cyclic type dependencies detected.\n")
        sys.stderr.write("  Logically this should not be possible.\n")
        sys.stderr.write(
            "  Emitted code may fail to compile due to ordering issues.\n")

      for emit_type in self.ordered_types:
        try:
          if isinstance(emit_type, types.StructType):
            self._emit_struct(hdr, emit_type)
          elif isinstance(emit_type, types.UnionType):
            self._emit_union(hdr, emit_type)
          elif isinstance(emit_type, types.WindowType):
            self._emit_window(hdr, emit_type)
          elif isinstance(emit_type, types.TypeAlias):
            self._emit_alias(hdr, emit_type)
        except ValueError as e:
          sys.stderr.write(f"Error emitting type '{emit_type}': {e}\n")
          hdr.write(f"// Unsupported type '{emit_type}': {e}\n\n")

      hdr.write(textwrap.dedent(f"""
    }} // namespace {system_name}
    """))


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
    # Use os.pathsep (';' on Windows, ':' on Unix) to avoid conflicts with
    # drive letters.
    conn = Context.default().connect("trace", f"-{os.pathsep}{args.file}")
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
