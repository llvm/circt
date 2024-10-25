import inspect
import argparse
import importlib.util
import sys
from pathlib import Path

def codegen2(bindings_package_name, path):
    file = import_module_from_path(Path(path))
    # print(module.context.dialects)
    # print(module.context.dialects['rtg'])
    # Iterate over all the members of the module

    rtg = importlib.import_module(bindings_package_name)
    with rtg.ir.Context() as ctx, rtg.ir.Location.unknown():
        rtg.register_dialects(ctx) 
        module = rtg.ir.Module.create()
        for name, obj in inspect.getmembers(file, inspect.isclass):
            # Check if the class has the custom decorator (the attribute _is_decorated)
            if hasattr(obj, '_is_scope') and obj._is_scope:
                # print(f"Instantiating {name}...")
                obj(module)  # Call the constructor
        return module

def codegen(module, path):
    file = import_module_from_path(Path(path))
    with module.context:
        for name, obj in inspect.getmembers(file, inspect.isclass):
            if hasattr(obj, '_is_scope') and obj._is_scope:
                obj(module)
        return m

def import_module_from_path(module_path):
    # Extract the module name from the path (useful if needed)
    module_name = module_path.stem  # Removes the '.py' part

    # Load the module dynamically
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    
    return module

# if __name__ == "__main__":
    # # Set up argument parser
    # parser = argparse.ArgumentParser(description="Driver.")
    # parser.add_argument("file", type=str, help="Path to the Python file to import")
    # parser.add_argument("--libpath", type=str, help="Python bindings path")
    # parser.add_argument("--libname", type=str, help="Python bindings module name")
    
    # # Parse command-line arguments
    # args = parser.parse_args()

    # if args.libpath:
    #     sys.path.append(args.libpath)

    # sys.path.append('/Users/martin.erhart/rtg/rtg-tblgen/build/tools/rtg-tblgen/python_packages/rtg_tblgen_core')

    # rtg = importlib.import_module(args.libname)
    # globals().update(vars(rtg.dialects))
    # globals().update(vars(rtg.ir))
    # import rtg_tblgen
    # from libmodule.dialects import (rv32i, rtg)
    # from libmodule.ir import *
    
    # Import the module

    # codegen(imported_module)
