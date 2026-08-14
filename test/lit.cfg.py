import lit.formats
import os

config.name = "ShaderMLIR"
config.test_format = lit.formats.ShTest(False)
config.suffixes = [".mlir"]
config.test_source_root = os.path.dirname(__file__)
config.test_exec_root = config.test_source_root

# Put LLVM bin dir at the front of PATH so FileCheck is found
llvm_bin = "/usr/lib/llvm-23/bin"
config.environment["PATH"] = llvm_bin + os.pathsep + config.environment["PATH"]

config.substitutions.append(("%PATH%", config.environment["PATH"]))
config.substitutions.append(("%shader-opt", os.path.join(config.shader_mlir_tools_dir, "shader-opt")))
