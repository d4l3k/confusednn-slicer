import copy
from collections import defaultdict


class GCodeBase:
    def __init__(self, args):
        self.args = args

    def __repr__(self):
        return f"{self.code} {' '.join(self.args)}"


class GCodePosition(GCodeBase):
    ARGS = "XYZEF"

    def __init__(self, args):
        super().__init__(args)

        self.positions = {}
        for arg in args:
            arg_type = arg[:1]
            assert arg_type in GCodeSetPosition.ARGS, f"invalid arg {arg}"
            self.positions[arg_type] = float(arg[1:])

    def update(self, position):
        self.positions.update(position.positions)

    def __getattr__(self, name: str) -> float:
        if name in GCodePosition.ARGS:
            return self.positions.get(name, None)
        return super().__getattr__(name)

    def __setattr__(self, name: str, value: float) -> None:
        if name in GCodePosition.ARGS:
            self.positions[name] = value
        super().__setattr__(name, value)

    def __repr__(self):
        args = [f"{key}{value}" for key, value in self.positions.items()]
        return f"{self.code} {' '.join(args)}"


class GCodeMove(GCodePosition):
    pass


class GCodeSetPosition(GCodePosition):
    pass


class GCodeSetFanSpeed(GCodeBase):
    def __init__(self, args):
        super().__init__(args)


def GCodeFanOff(args):
    assert len(args) == 0, f"should not have args {args}"
    return GCodeSetFanSpeed(["S0"])


GCODES = {
    "G1": GCodeMove,
    "G92": GCodeSetPosition,
    "M106": GCodeSetFanSpeed,
    "M107": GCodeFanOff,
}

for code, cls in GCODES.items():
    cls.code = code


class GCode:
    def __init__(self):
        self.commands = []

    @classmethod
    def from_file(cls, filename):
        gcode = GCode()
        with open(filename, "r") as fh:
            for line_str in fh.readlines():
                code = line_str.split(";")[0].strip()
                if not code:
                    continue
                gcode.parse(code)
        return gcode

    def parse(self, code):
        command, *args = code.split(" ")
        if command not in GCODES:
            # print(f"unsupported command {command} {args}. skipping")
            return
        gcode = GCODES[command](args)
        self.commands.append(gcode)

    def __repr__(self):
        commands = "\n" + "\n".join(f"  {item}" for item in self.commands) + "\n"
        return f"GCode(commands=[{commands}])"

    def normalize(self):
        gcode = GCode()
        pos = GCodePosition(["E0", "X0", "Y0"])
        for cmd in self.commands:
            if isinstance(cmd, GCodeSetPosition):
                pos.update(pos)
            elif isinstance(cmd, GCodeMove):
                new = GCodeMove([])
                new.update(pos)
                new.update(cmd)
                if cmd.E:
                    new.E = cmd.E - pos.E
                else:
                    new.E = 0
                pos.update(cmd)
                if cmd.X or cmd.Y or cmd.E:
                    assert (
                        new.X is not None
                        and new.Y is not None
                        and new.E is not None
                        and new.F is not None
                    ), f"missing field {new} from {cmd}"
                    # print(new, cmd)
                    gcode.commands.append(new)

        return gcode

    def split_layers(self):
        normalized = self.normalize()
        layers = defaultdict(lambda: [])
        for command in normalized.commands:
            assert isinstance(command, GCodeMove)
            layers[command.Z].append(command)

        return layers


if __name__ == "__main__":
    gcode = GCode.from_file("data/DSA_1u.gcode")
    normalized = gcode.normalize()
    assert len(normalized.commands)
    layers = gcode.split_layers()
    assert len(layers) > 0
    for height, commands in layers.items():
        print(height, len(commands))
