#  Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
#  See https://llvm.org/LICENSE.txt for license information.
#  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from ._esi_ops_gen import *
from mlir._mlir_libs._circt._esi import *

from io import FileIO
import json
import pathlib
import re
from typing import Dict, List, Optional, Set, Tuple

__dir__ = pathlib.Path(__file__).parent


def _camel_to_snake(camel: str):
  return re.sub(r'(?<!^)(?=[A-Z])', '_', camel).lower()


class SoftwareApiBuilder:

  class Module:

    def __init__(self, name: str):
      self.name = name
      self.instances: Dict[str, SoftwareApiBuilder.Module] = {}
      self.services: List[Dict] = []

  def __init__(self, services_json: str, capnp_schema: Optional[str]):
    self.services = json.loads(services_json)
    self.cosim_schema = capnp_schema
    self.types: Dict[str, Dict] = {}
    self.modules: Dict[str, SoftwareApiBuilder.Module] = {}

  def _get_module(self, mod_sym: str):
    if mod_sym not in self.modules:
      self.modules[mod_sym] = SoftwareApiBuilder.Module(mod_sym)
    return self.modules[mod_sym]

  def build(self, os: FileIO):
    self.os = os
    self._write_header()
    for decl in self.services["declarations"]:
      self._write_decl(decl)
    os.write("\n\n")

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

    for mod in self.services["modules"]:
      m = self._get_module(mod["symbol"])
      for svc in mod["services"]:
        m.services.append(svc)

    self._write_namespace("modules")
    for mod in self.modules.values():
      self._write_module(mod)
    os.write("\n\n")

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

  def get_top_levels(self):
    return [
        top_level["module"][1:] for top_level in self.services["top_levels"]
    ]

  def get_services(self, top_level: str):
    for top_level in self.services["top_levels"]:
      if top_level["module"][1:] != top_level:
        continue
      for service in top_level["services"]:
        yield service

  def get_type_name(self, type: Dict):
    if "capnp_name" in type:
      name = type["capnp_name"]
    else:
      name = "".join([c if c.isalnum() else '_' for c in type["mlir_name"]])
    self.types[name] = type
    return name


class PythonApiBuilder(SoftwareApiBuilder):

  def __init__(self, services_json: str, capnp_schema: Optional[str]):
    super().__init__(services_json, capnp_schema)
    self.wrote_types = False

  def build(self, system_name: str, output_dir: pathlib.Path):
    libdir = output_dir / "esi_rt"
    if not libdir.exists():
      libdir.mkdir()
    init_file = libdir / "__init__.py"
    init_file.touch()
    common_file = libdir / "common.py"
    common_file.write_text((__dir__ / "esi_runtime_common.py").read_text())
    main = libdir / f"{system_name}.py"
    super().build(main.open("w"))

  def _write_header(self):
    self._writeline("from .common import *")
    self._writeline()
    self._writeline()

  def _write_namespace(self, namespace: str):
    if namespace == "modules":
      self._writeline("class DesignModules:\n")

  def _write_decl(self, decl: Dict):
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
    write = self._writeline

    write(f"  class {mod.name}:")
    write()
    write("    def __init__(self):")
    for inst_name, child_mod in mod.instances.items():
      inst_snake = _camel_to_snake(inst_name)
      write(f"      self.{inst_snake} = DesignModules.{child_mod.name}()")

    for svc in mod.services:
      clients = svc['clients']
      if len(clients) == 0:
        continue
      if len(mod.instances) > 0:
        write()

      ports = {}
      for client in clients:
        port = client['port']['inner']
        if port not in ports:
          ports[port] = []
        ports[port].append(client)

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
      svc_name = svc["service"]
      svc_name_snake_case = _camel_to_snake(svc_name)
      write(f"      self.{svc_name_snake_case} = {svc_name}(" +
            ", ".join([f"{pn}_ports={pn}" for pn in ports.keys()]) + ")")
    write()

  def _write_top(self, top: SoftwareApiBuilder.Module):
    write = self._writeline
    write(f"{_camel_to_snake(top.name)} = DesignModules.{top.name}()")

  def _write_type(self, name: str, type: Dict):
    write = self._writeline
    if not self.wrote_types:
      self.wrote_types = True
      write("class ESITypes:\n")

    def get_python_type(type: Dict):
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
