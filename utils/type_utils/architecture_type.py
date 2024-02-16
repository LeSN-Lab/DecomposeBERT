import enum


class ArchitectureType(enum.Enum):
    only_encoder = 0
    only_decoder = 1
    both = 2

    @staticmethod
    def get_architecture_type(arc_type):
        if arc_type == "only_encoder":
            return ArchitectureType.only_encoder
        elif arc_type == "only_decoder":
            return ArchitectureType.only_decoder
        else:
            return ArchitectureType.both
