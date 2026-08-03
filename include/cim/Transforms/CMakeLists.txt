set(LLVM_TARGET_DEFINITIONS Passes.td)
mlir_tablegen(Passes.h.inc -gen-pass-decls -name CIM)
add_public_tablegen_target(MLIRCIMTransformsIncGen)

add_mlir_doc(Passes CIMPasses Passes/ -gen-pass-doc)
