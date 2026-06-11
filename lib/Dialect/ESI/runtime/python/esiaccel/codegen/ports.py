#  Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
#  See https://llvm.org/LICENSE.txt for license information.
#  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""Port-kind strategy table for the C++ module generator.

Centralises everything that varies between the ESI port kinds (function,
callback, to-host / from-host channel, MMIO region, telemetry metric, plain
bundle) into a single `CppPortKind` record per kind plus two pure rendering
functions, so the generator drives every port kind from one table. The
renderers are pure functions of a `CppPortKind`, so they are golden-tested
without a live accelerator.
"""

from dataclasses import dataclass, field as _dc_field
from typing import List, Optional, Tuple

from ..types import (BundlePort as _BundlePort, FunctionPort as _FunctionPort,
                     CallbackPort as _CallbackPort, ToHostPort as _ToHostPort,
                     FromHostPort as _FromHostPort, MMIORegion as
                     _MMIORegionPort, MetricPort as _MetricPort)


@dataclass
class CppPortGroup:
  """All strings needed to emit one port slot (member + ctor + find) in Connected."""
  struct_decls: List[str] = _dc_field(default_factory=list)
  member_decl: str = ""
  ctor_params: List[str] = _dc_field(default_factory=list)
  init_entry: str = ""
  find_code: str = ""
  make_unique_args: List[str] = _dc_field(default_factory=list)
  post_connect: str = ""
  using_aliases: List[Tuple[str, str]] = _dc_field(default_factory=list)


@dataclass(frozen=True)
class CppPortKind:
  """Per-port-kind constants for generating a module's typed port slots.

  Holds everything that varies between port kinds in one record.
  `member_template` (typed channel/func ports) and `member_ref` (service /
  bundle ports) are mutually exclusive. `find_style` / `indexed_find` select
  the `connect()` resolution snippet for scalar and indexed ports
  respectively; `alias_kind` selects which `using` aliases to emit for the
  typed template parameters.
  """
  service_type: Optional[
      str]  # findPortAs<...> service type (None = plain bundle)
  find_style: str  # 'ptr' | 'read_chan' | 'write_chan' | 'deref' | 'bundle'
  indexed_find: str  # 'as_ptr' | 'chan_read' | 'chan_write' | 'bundle_addr'
  param_type: str  # ctor parameter type
  param_suffix: str  # '_port' | '_chan' | '_svc'
  member_template: Optional[
      str]  # typed member, e.g. "esi::TypedFunction<{p}Args, {p}Result>"
  member_ref: Optional[str]  # non-typed member / ctor ref type
  indexed_elem: Optional[
      str]  # IndexedPorts<...> element (None => use member_template)
  alias_kind: Optional[str]  # None | 'func' (Args/Result) | 'chan' (Data)
  connectable: bool  # whether generated connect() calls .connect()

  def member_type(self, alias_prefix: Optional[str] = None) -> str:
    """Return the C++ member type string for this kind (no member name).

    For typed ports (function/callback/to-host/from-host channels)
    `alias_prefix` is required; the returned template parameters are written
    using the alias names (`<prefix>Args`, `<prefix>Result`, `<prefix>Data`)
    that should be emitted at module-class scope. For non-typed ports (MMIO
    regions, telemetry metrics, plain bundles) `alias_prefix` is ignored and
    the runtime reference/pointer type is returned directly.
    """
    if self.member_template is not None:
      assert alias_prefix is not None, (
          "alias_prefix is required for typed ports to avoid emitting long "
          "mangled type names inline (which would also collide across "
          "modules' `using` declarations)")
      return self.member_template.format(p=alias_prefix)
    # Every non-typed kind defines a reference member type.
    assert self.member_ref is not None
    return self.member_ref

  def indexed_elem_type(self, alias_prefix: Optional[str] = None) -> str:
    """Return the storage type used inside an `IndexedPorts<T>` for this kind.

    Typed ports use the same `TypedFunction<...>` / `TypedReadPort<...>` etc.
    that `member_type` produces. MMIO regions, telemetry metrics, and plain
    bundle ports are stored as raw pointers because `std::map<int, T&>` is
    ill-formed.
    """
    if self.indexed_elem is not None:
      return self.indexed_elem
    return self.member_type(alias_prefix=alias_prefix)

  def scalar_find_code(self, v: str, appid_expr: str) -> str:
    """Render the connect()-body snippet that resolves one scalar port into the
    local `v`. Pure function of the kind so it can be golden-tested without a
    live accelerator (see `test_port_find_code_golden`)."""
    if self.find_style == "ptr":
      return (f"auto *{v} =\n"
              f"    esi::findPortAsOrThrow<{self.service_type}>(\n"
              f"        rawModule, {appid_expr});")
    if self.find_style == "read_chan":
      return (f"auto &{v} =\n"
              f"    esi::findPortAsOrThrow<{self.service_type}>(\n"
              f'        rawModule, {appid_expr})->getRawRead("data");')
    if self.find_style == "write_chan":
      return (f"auto &{v} =\n"
              f"    esi::findPortAsOrThrow<{self.service_type}>(\n"
              f'        rawModule, {appid_expr})->getRawWrite("data");')
    if self.find_style == "deref":
      return (f"auto &{v} =\n"
              f"    *esi::findPortAsOrThrow<{self.service_type}>(\n"
              f"        rawModule, {appid_expr});")
    if self.find_style == "bundle":
      # plain BundlePort
      return f"auto &{v} = esi::findPortOrThrow(rawModule, {appid_expr});"
    raise ValueError(f"unknown find_style: {self.find_style!r}")

  def indexed_find_code(self, map_var: str, appid_expr: str) -> str:
    """Render the per-index `try_emplace` body for an `IndexedPorts<T>` group.
    Pure function of the kind (golden-tested alongside `scalar_find_code`)."""
    if self.indexed_find == "as_ptr":
      return (f"  {map_var}.try_emplace(\n"
              f"      static_cast<int>(idx),\n"
              f"      esi::findPortAsOrThrow<{self.service_type}>(\n"
              f"          rawModule, {appid_expr}));")
    if self.indexed_find == "chan_read":
      return (f"  auto *svc =\n"
              f"      esi::findPortAsOrThrow<{self.service_type}>(\n"
              f"          rawModule, {appid_expr});\n"
              f"  {map_var}.try_emplace(\n"
              f"      static_cast<int>(idx),\n"
              f'      svc->getRawRead("data"));')
    if self.indexed_find == "chan_write":
      return (f"  auto *svc =\n"
              f"      esi::findPortAsOrThrow<{self.service_type}>(\n"
              f"          rawModule, {appid_expr});\n"
              f"  {map_var}.try_emplace(\n"
              f"      static_cast<int>(idx),\n"
              f'      svc->getRawWrite("data"));')
    if self.indexed_find == "bundle_addr":
      return (f"  {map_var}.try_emplace(\n"
              f"      static_cast<int>(idx),\n"
              f"      &esi::findPortOrThrow(rawModule, {appid_expr}));")
    raise ValueError(f"unknown indexed_find: {self.indexed_find!r}")


# Ordered list of (runtime port class, kind). `cpp_port_kind` returns the first
# match, falling back to `CPP_BUNDLE_KIND` for any unrecognised service port.
CPP_PORT_KINDS: List[Tuple[type, CppPortKind]] = [
    (_FunctionPort,
     CppPortKind(service_type="esi::services::FuncService::Function",
                 find_style="ptr",
                 indexed_find="as_ptr",
                 param_type="esi::services::FuncService::Function *",
                 param_suffix="_port",
                 member_template="esi::TypedFunction<{p}Args, {p}Result>",
                 member_ref=None,
                 indexed_elem=None,
                 alias_kind="func",
                 connectable=True)),
    (_CallbackPort,
     CppPortKind(service_type="esi::services::CallService::Callback",
                 find_style="ptr",
                 indexed_find="as_ptr",
                 param_type="esi::services::CallService::Callback *",
                 param_suffix="_port",
                 member_template="esi::TypedCallback<{p}Args, {p}Result>",
                 member_ref=None,
                 indexed_elem=None,
                 alias_kind="func",
                 connectable=False)),
    (_ToHostPort,
     CppPortKind(service_type="esi::services::ChannelService::ToHost",
                 find_style="read_chan",
                 indexed_find="chan_read",
                 param_type="esi::ReadChannelPort &",
                 param_suffix="_chan",
                 member_template="esi::TypedReadPort<{p}Data>",
                 member_ref=None,
                 indexed_elem=None,
                 alias_kind="chan",
                 connectable=True)),
    (_FromHostPort,
     CppPortKind(service_type="esi::services::ChannelService::FromHost",
                 find_style="write_chan",
                 indexed_find="chan_write",
                 param_type="esi::WriteChannelPort &",
                 param_suffix="_chan",
                 member_template="esi::TypedWritePort<{p}Data>",
                 member_ref=None,
                 indexed_elem=None,
                 alias_kind="chan",
                 connectable=True)),
    (_MMIORegionPort,
     CppPortKind(service_type="esi::services::MMIO::MMIORegion",
                 find_style="deref",
                 indexed_find="as_ptr",
                 param_type="esi::services::MMIO::MMIORegion &",
                 param_suffix="_svc",
                 member_template=None,
                 member_ref="esi::services::MMIO::MMIORegion &",
                 indexed_elem="esi::services::MMIO::MMIORegion *",
                 alias_kind=None,
                 connectable=False)),
    (_MetricPort,
     CppPortKind(service_type="esi::services::TelemetryService::Metric",
                 find_style="deref",
                 indexed_find="as_ptr",
                 param_type="esi::services::TelemetryService::Metric &",
                 param_suffix="_svc",
                 member_template=None,
                 member_ref="esi::services::TelemetryService::Metric &",
                 indexed_elem="esi::services::TelemetryService::Metric *",
                 alias_kind=None,
                 connectable=True)),
]

# Fallback for any service port that doesn't match a standard specialization
# (e.g. a custom `@esi.ServiceDecl`-defined service).
CPP_BUNDLE_KIND = CppPortKind(service_type=None,
                              find_style="bundle",
                              indexed_find="bundle_addr",
                              param_type="esi::BundlePort &",
                              param_suffix="_port",
                              member_template=None,
                              member_ref="esi::BundlePort &",
                              indexed_elem="esi::BundlePort *",
                              alias_kind=None,
                              connectable=False)


def cpp_port_kind(port: _BundlePort) -> CppPortKind:
  """Return the `CppPortKind` describing `port`, or `CPP_BUNDLE_KIND` as fallback."""
  for cls, kind in CPP_PORT_KINDS:
    if isinstance(port, cls):
      return kind
  return CPP_BUNDLE_KIND
