import os

import lit.formats
from lit.llvm import llvm_config

config.name = "CIM-MLIR"
config.test_format = lit.formats.ShTest(not llvm_config.use_lit_shell)

config.suffixes = [".mlir"]
config.test_source_root = os.path.dirname(__file__)
config.test_exec_root = os.path.join(config.cim_obj_root, "test")

llvm_config.use_default_substitutions()

config.excludes = ["lit.cfg.py", "lit.site.cfg.py", "CMakeLists.txt"]

tool_dirs = [config.cim_tools_dir, config.llvm_tools_dir]
tools = ["cim-opt", "FileCheck"]

llvm_config.add_tool_substitutions(tools, tool_dirs)
