#  Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
#  See https://llvm.org/LICENSE.txt for license information.
#  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from ._esi_ops_gen import *
from mlir._mlir_libs._circt._esi import *

from io import FileIO
import json
import pathlib
import re
from typing import Dict, List, Optional

__dir__ = pathlib.Path(__file__).parent


def _camel_to_snake(camel: str):
  return re.sub(r'(?<!^)(?=[A-Z])', '_', camel).lower()


class SoftwareApiBuilder:
  """Parent class for all software API builders. Defines an interfaces and tries
  to encourage code sharing and API consistency (between languages)."""

  class Module:
    """Bookkeeping about modules."""

    def __init__(self, name: str):
      self.name = name
      self.instances: Dict[str, SoftwareApiBuilder.Module] = {}
      self.services: List[Dict] = []

  def __init__(self, services_json: str, capnp_schema: Optional[str]):
    """Read in the system descriptor and set up bookkeeping structures."""
    self.services = json.loads(services_json)
    self.cosim_schema = capnp_schema
    self.types: Dict[str, Dict] = {}
    self.modules: Dict[str, SoftwareApiBuilder.Module] = {}

    # Get all the modules listed in the service hierarchy. Populate their
    # 'instances' properly.
    for top in self.services["top_levels"]:
      top_mod = self._get_module(top["module"][1:])
      for svc in top["services"]:
        parent: SoftwareApiBuilder.Module = top_mod
        for inner_ref in [
            (inst["outer_sym"], inst["inner"]) for inst in svc["instance_path"]
        ]:
          m = self._get_module(inner_ref[0])
          parent.instances[inner_ref[1]] = m
          parent = m

    # For any modules which have services, add them as appropriate.
    for mod in self.services["modules"]:
      m = self._get_module(mod["symbol"])
      for svc in mod["services"]:
        m.services.append(svc)

  def _get_module(self, mod_sym: str):
    """Get a module adding an entry if it doesn't exist."""
    if mod_sym not in self.modules:
      self.modules[mod_sym] = SoftwareApiBuilder.Module(mod_sym)
    return self.modules[mod_sym]

  def build(self, os: FileIO):
    """Output the API (in a pre-determined order) via callbacks. Encourages some
    level of consistency between language APIs."""

    self.os = os
    self._write_header()

    for decl in self.services["declarations"]:
      self._write_decl(decl)
    os.write("\n\n")

    self._write_namespace("modules")
    for mod in self.modules.values():
      self._write_module(mod)
    os.write("\n\n")

    # Need to write the modules first since it populates the types set as a side
    # effect.
    self._write_namespace("types")
    for type_name, type_dict in self.types.items():
      self._write_type(type_name, type_dict)

    self._writeline()
    self._writeline()
    for top in self.services["top_levels"]:
      self._write_top(self._get_module(top["module"][1:]))
    self.os = None

  def _writeline(self, line: str = ""):
    self.os.write(line + "\n")

  def _write_header(self):
    pass

  def _write_namespace(self, namespace: str):
    pass

  def _write_decl(self, decl: Dict):
    assert False, "Unimplemented"

  def _write_module(self, mod):
    # mod is SoftwareApiBuilder.Module.
    assert False, "Unimplemented"

  def _write_type(self, name: str, type: Dict):
    assert False, "Unimplemented"

  def _write_top(self, top: Dict):
    assert False, "Unimplemented"

  def get_type_name(self, type: Dict):
    """Create a name for 'type', record it, and return it."""
    if "capnp_name" in type:
      name = type["capnp_name"]
    else:
      name = "".join([c if c.isalnum() else '_' for c in type["mlir_name"]])
    self.types[name] = type
    return name


class PythonApiBuilder(SoftwareApiBuilder):

  def __init__(self, services_json: str, capnp_schema: Optional[str]):
    super().__init__(services_json, capnp_schema)

  def build(self, system_name: str, output_dir: pathlib.Path):
    """Emit a Python ESI runtime library into 'output_dir'."""
    libdir = output_dir / "esi_rt"
    if not libdir.exists():
      libdir.mkdir()

    # Create __init__.py and copy in the standard files.
    init_file = libdir / "__init__.py"
    init_file.touch()
    common_file = libdir / "common.py"
    common_file.write_text((__dir__ / "esi_runtime_common.py").read_text())

    # Emit the system-specific API.
    main = libdir / f"{system_name}.py"
    super().build(main.open("w"))

  def _write_header(self):
    self._writeline("from .common import *")
    self._writeline()
    self._writeline()

  def _write_namespace(self, namespace: str):
    """We 'namespace' the API with classes."""
    write = self._writeline
    if namespace == "modules":
      write("class DesignModules:\n")
    elif namespace == "types":
      write("class ESITypes:\n")

  def _write_decl(self, decl: Dict):
    """Emit a ServiceDeclaration. Example:
    ```
    class HostComms:

      def __init__(self, to_host_ports: typing.List[Port], from_host_ports: typing.List[Port], req_resp_ports: typing.List[Port]):
        self.to_host = to_host_ports
        self.from_host = from_host_ports
        self.req_resp = req_resp_ports

      def to_host_read_any(self):
        for p in self.to_host:
          rc = p.read(block=False)
          if rc is not None:
            return rc
        return None

      def req_resp_read_any(self):
        for p in self.req_resp:
          rc = p.read(block=False)
          if rc is not None:
            return rc
        return None
    ```
    """

    os = self.os
    write = self._writeline

    write(f"class {decl['name']}:")
    write()
    os.write("  def __init__(self, ")
    os.write(", ".join(
        [f"{port['name']}_ports: typing.List[Port]" for port in decl['ports']]))
    write("):")
    for port in decl["ports"]:
      write(f"    self.{port['name']} = {port['name']}_ports")

    for port in decl["ports"]:
      pn = port['name']
      if "to-server-type" in port:
        write(f"""
  def {pn}_read_any(self):
    for p in self.{pn}:
      rc = p.read(block=False)
      if rc is not None:
        return rc
    return None""")

  def _write_module(self, mod: SoftwareApiBuilder.Module):
    """Emit a module class for 'mod'. Examples (incl. the 'namespace'):
    ```
    class DesignModules:

      class Top:

        def __init__(self):
          self.mid = DesignModules.Mid()

      class Mid:

        def __init__(self):
          from_host = [
              WritePort(['Producer', 'loopback_in'], read_type=None, write_type=ESITypes.I32),
            ]
          to_host = [
              ReadPort(['Consumer', 'loopback_out'], read_type=ESITypes.I32, write_type=None),
            ]
          req_resp = [
              WritePort(['LoopbackInOut', 'loopback_inout'], read_type=None, write_type=ESITypes.I32),
              ReadPort(['LoopbackInOut', 'loopback_inout'], read_type=ESITypes.I16, write_type=None),
            ]
          self.host_comms = HostComms(from_host_ports=from_host, to_host_ports=to_host, req_resp_ports=req_resp)
    ```
    """

    write = self._writeline

    write(f"  class {mod.name}:")
    write()
    write("    def __init__(self):")
    # Emit the instances contained in this module.
    for inst_name, child_mod in mod.instances.items():
      inst_snake = _camel_to_snake(inst_name)
      write(f"      self.{inst_snake} = DesignModules.{child_mod.name}()")

    # Emit any services which are instantiated by this module.
    for svc in mod.services:
      clients = svc['clients']
      if len(clients) == 0:
        continue
      if len(mod.instances) > 0:
        write()

      # Assemble lists of clients for each service port.
      ports = {}
      for client in clients:
        port = client['port']['inner']
        if port not in ports:
          ports[port] = []
        ports[port].append(client)

      # For each port, assemble a list of 'client' ports and create a 'Port' for
      # each one.
      for port_name, port_clients in ports.items():
        write(f"      {port_name} = [")
        for pclient in port_clients:
          if "to_client_type" in pclient and "to_server_type" in pclient:
            read_type_name = "ESITypes." + self.get_type_name(
                pclient["to_server_type"])
            write_type_name = "ESITypes." + self.get_type_name(
                pclient["to_client_type"])
            port_class = "ReadWritePort"
          elif "to_client_type" in pclient:
            read_type_name = "None"
            write_type_name = "ESITypes." + self.get_type_name(
                pclient["to_client_type"])
            port_class = "WritePort"
          elif "to_server_type" in pclient:
            read_type_name = "ESITypes." + self.get_type_name(
                pclient["to_server_type"])
            write_type_name = "None"
            port_class = "ReadPort"
          else:
            continue

          write(f"          {port_class}({repr(pclient['client_name'])}," +
                f" read_type={read_type_name}," +
                f" write_type={write_type_name}),")
        write("        ]")

      # Instantiate the service with the lists of service ports.
      svc_name = svc["service"]
      svc_name_snake_case = _camel_to_snake(svc_name)
      write(f"      self.{svc_name_snake_case} = {svc_name}(" +
            ", ".join([f"{pn}_ports={pn}" for pn in ports.keys()]) + ")")
    write()

  def _write_top(self, top: SoftwareApiBuilder.Module):
    """Write top-level instantiations of the modules."""
    write = self._writeline
    write(f"{_camel_to_snake(top.name)} = DesignModules.{top.name}()")

  def _write_type(self, name: str, type: Dict):
    """Write a type. Example: `I32 = IntType(32, False)`"""
    write = self._writeline

    def get_python_type(type: Dict):
      """Get a Python code string instantiating 'type'."""
      if type["dialect"] == "esi" and type["mnemonic"] == "channel":
        return get_python_type(type["inner"])
      if type["dialect"] == "builtin":
        m: str = type["mnemonic"]
        if m.startswith("i") or m.startswith("ui"):
          width = int(m.strip("ui"))
          return f"IntType({width}, False)"
        if m.startswith("i") or m.startswith("si"):
          width = int(m.strip("si"))
          return f"IntType({width}, True)"
      assert False, "unimplemented type"

    write(f"  {name} = {get_python_type(type['type_desc'])}")
    return
