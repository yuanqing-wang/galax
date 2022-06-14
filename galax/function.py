"""Built-in functions."""
from collections import namedtuple
ReduceFunction = namedtuple(
    "ReduceFunction",
    ["op", "msg_field", "out_field"]
)
