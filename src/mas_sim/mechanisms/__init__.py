from .cbba import CBBA
from .cnp import ContractNet
from .greedy import Greedy
from .nearest import Nearest
from .round_robin import RoundRobin
from .ssi import SequentialSingleItem

MECHANISMS = {
    "round_robin": RoundRobin, "nearest": Nearest, "greedy": Greedy,
    "cnp": ContractNet, "ssi": SequentialSingleItem, "cbba": CBBA,
}

LABELS = {"round_robin": "Round-robin", "nearest": "Nearest-agent",
          "greedy": "Greedy centralized", "cnp": "Contract Net", "ssi": "SSI auction", "cbba": "CBBA"}
