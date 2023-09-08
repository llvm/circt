from __future__ import annotations

from pathlib import Path


import circt
import queue
from circt.dialects.om import Evaluator, Object
from circt.ir import Context, Module
import argparse
import json

class GenOmir:
  def __init__(self) -> None:
    self.objectModelList = [] 
    self.objectIdMap = {}
    self.objId = 0
    self.nodeList = queue.Queue()

  def load_module_from_file(self, path: Path) -> Module:
    """This functions loads an MLIR Module from a file at the given Path."""
    with Context() as ctx:
        circt.register_dialects(ctx)
        with open(path) as text:
            module = Module.parse(text.read())

    return module

  def getId(self, obj : Object):
      id = self.objectIdMap.get(obj)
      if id is None:
        id = self.objId
        self.objectIdMap[obj] = id
        self.objId += 1
      return id
    
  def addDataForObj(self, obj: Object):
      nodeOmir = {} 
      nodeOmir["id"] = "OMID:" + str(self.getId(obj))
      nodeFields = []
      for (field_name, data) in obj:
          nodeField = {}
          nodeField["name"] = field_name
          if isinstance(data, Object):
            self.nodeList.put(data)
            nodeField["value"] = "OMReference:" + str( self.getId(data))
          else:
            nodeField["value"] = data
          nodeFields.append(nodeField)
      nodeOmir["fields"] = nodeFields
      return nodeOmir

  def get_objects_from_container(self, obj: Object) -> list[Object]:
      """Implement the "abi" for OMNode objects.

      For the top level container, look for fields that are an object of class
      OMIR. The linker may have mangled names, so just check if OMIR is in the
      class name at all, which is sketchy but works.

      Each of those object fields will themselves have object fields
      representing every OMNode. Collect and return those objects.
      """
      objects = []
      for (field_name, data) in obj:
          if not isinstance(data, Object):
            continue
          if not "OMIR" in str(data.type):
            continue
          for (_, om_node) in data:
            if not isinstance(om_node, Object):
              continue
            self.nodeList.put(om_node)

  def run(self, path: Path, mainClass: String, args: [Any]):
    module = self.load_module_from_file(path)
    evaluator = Evaluator(module)
    root  = evaluator.instantiate(mainClass, *args)
    self.get_objects_from_container(root)
    while not self.nodeList.empty():
        node = self.nodeList.get()
        self.objectModelList.append(self.addDataForObj(node))
    json_object = json.dumps(self.objectModelList, indent=4)
    return json_object


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('path_argument', type=Path, help='The mlir file path to process.')
    parser.add_argument('output_json', type=Path, help='The output omir json file path.')
    parser.add_argument('--main', help='The name of the main OM class',
                        required=True,
                        dest='mainClass')
    parser.add_argument('--args',
                        help='Arguments to pass to instantiate the main OM class',
                        required=False,
                        dest='mainArgs',
                        action='append',
                        default=[])

    args = parser.parse_args()
    path_argument = args.path_argument

    obj = GenOmir()
    omJson = obj.run(path_argument, args.mainClass, args.mainArgs)

    output_file_path = args.output_json

    try:
        # Open the output file for writing
        with open(output_file_path, 'w') as output_file:
            output_file.write(omJson)
        print(f"File '{output_file_path}' has been created successfully.")
    except Exception as e:
        print(f"Error: {e}")

