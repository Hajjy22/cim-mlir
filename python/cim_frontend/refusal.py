"""The one way this front end declines to compile something.

Every refusal is an exception carrying a message that names the offending
ONNX node or tensor and states the rule it broke. The alternative -- a
warning plus a best-effort import -- is the failure this project exists to
avoid: silently dropping an op the importer does not understand produces a
module that compiles cleanly, runs, and computes something other than the
model. A wrong number that looks right is worse than a refusal.

This mirrors what the passes themselves do. cim-partition refuses a plain
linalg.matmul rather than guessing a transpose; cim-legalize-precision
refuses to invent calibration data. Same discipline, one level further
out.
"""


class Refusal(Exception):
    """This model, or this part of it, is outside the v0.1 contract.

    `where` names the ONNX node or tensor, so the message can point at
    something the user can find in their own model rather than at our
    internal state.
    """

    def __init__(self, message, where=None):
        self.where = where
        self.message = message
        super().__init__(self.format())

    def format(self):
        prefix = "cim-import-onnx: "
        if self.where:
            return f"{prefix}{self.where}: {self.message}"
        return f"{prefix}{self.message}"
