/* driver.c -- the C side of the opt-in real-binary verification.
 *
 * run_check is the compiled MLIR function: cim-lower-to-target's output
 * taken through mlir-opt's --convert-func-to-llvm, mlir-translate, and a
 * real linker (see test/real-target/CMakeLists.txt). It takes no arguments
 * and returns nothing -- the check itself is a cf.assert baked into the
 * MLIR source, which becomes a genuine SIGABRT if it fires. This driver's
 * only job is to be a `main` that calls it and returns 0 if it didn't
 * abort; there is nothing left for C to check.
 */
extern void run_check(void);

int main(void) {
  run_check();
  return 0;
}
