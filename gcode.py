import copy
from collections import defaultdict

import math
import numpy as np


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

    def distance(self, b):
        return math.sqrt(
            (self.X-b.X)**2 + (self.Y-b.Y)**2 + (self.Z-b.Z)**2
        )

    def update(self, position):
        self.positions.update(position.positions)

    def __eq__(self, other):
        return self.positions == other.positions

    def __hash__(self):
        return hash(self.__repr__())

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
                    assert not len(gcode.commands) or gcode.commands[-1] != new, f"duplicate command {new}"
                    gcode.commands.append(new)

        return gcode

    def split_layers(self):
        assert self.normalized(), "must be normalized"

        layers = defaultdict(lambda: GCode())
        for command in self.commands:
            assert isinstance(command, GCodeMove)
            layers[command.Z].commands.append(command)

        return layers

    def normalized(self):
        return all(isinstance(cmd, GCodeMove) for cmd in self.commands)

    def path_length(self):
        if len(self.commands) == 0:
            return 0.0
        assert self.normalized(), "must be normalized"
        computed_len = 0.0
        start = self.commands[0]
        for cmd in self.commands[1:]:
            computed_len += start.distance(cmd)
            start = cmd
        return computed_len

    def rasterize(self, n):
        assert self.normalized(), "must be normalized"
        assert len(self.commands) > 1, "can't rasterize single point"
        step_size = (len(self.commands)-1) / (n-1)
        samples = defaultdict(lambda: 0)
        targets = [i * step_size for i in range(n-1)]
        targets.append(len(self.commands) - 1)
        assert len(targets) == n, f"{targets}, {len(self.commands)}, {step_size}"
        for i in targets:
            samples[int(i)] += 1

        idxs = list(samples.keys())
        idxs.sort()

        gcode = GCode()
        for i in idxs:
            count = samples[i]
            cmd = copy.deepcopy(self.commands[i])
            cmd.E /= count
            gcode.commands.append(cmd)
            if count > 1:
                next = copy.deepcopy(self.commands[i+1])
                xstep = (next.X-cmd.X) / count
                ystep = (next.Y-cmd.Y) / count
                zstep = (next.Z-cmd.Z) / count
                for step in range(1, count):
                    point = copy.deepcopy(cmd)
                    point.X += xstep * step
                    point.Y += ystep * step
                    point.Z += zstep * step
                    gcode.commands.append(point)

        assert len(gcode.commands) == n, f"wrong number of outputs {n} != {len(gcode.commands)}: {gcode}"
        assert self.commands[0].distance(gcode.commands[0]) < 0.0001,  "must include first point"
        assert self.commands[-1].distance(gcode.commands[-1]) < 0.0001, "must include last point"

        return gcode




if __name__ == "__main__":
    gcode = GCode.from_file("data/DSA_1u.gcode")
    normalized = gcode.normalize()
    assert len(normalized.commands)
    assert normalized.path_length() > 0, "missing path"
    assert normalized.rasterize(10)
    layers = normalized.split_layers()
    assert len(layers) > 0
    for height, gcode in layers.items():
        assert height and len(gcode.commands) > 0

    rasterized = gcode.rasterize(200)
    assert len(rasterized.commands) > len(gcode.commands), f"{len(gcode.commands)}"
    command_counts = defaultdict(lambda: 0)
    for cmd in rasterized.commands:
        command_counts[cmd] += 1

    for cmd, count in command_counts.items():
        if count > 1:
            print([(i, b) for i, b in enumerate(rasterized.commands) if b == cmd])
