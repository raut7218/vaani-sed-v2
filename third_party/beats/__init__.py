"""BEATs, vendored from https://github.com/microsoft/unilm/tree/master/beats (MIT).

Local modification: the upstream files use flat imports (`from backbone import ...`),
which only resolve when that directory is itself on sys.path. They are rewritten to
relative imports (`from .backbone import ...`) so the code works when imported as a
package. Nothing else is changed.
"""
