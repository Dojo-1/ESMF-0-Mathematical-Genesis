"""G0-AH: bounded dendritic-compartment reference kernel, not a biological brain.

Discrete, event-sourced operator model, NOT the G0-AG joint CTMC. The
neuroscience-inspired operators are H, geometric coordinate adapters A, and
conservation properties P conditional on these synthetic bookkeeping rules.
No independent semantic, biological, or efficiency advantage is claimed.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
import math

Q = tuple[float, float, float, float]
ONE: Q = (1., 0., 0., 0.)
CAP = 3
DIRECTIONS = ((0, 0, 1), (0, 1, 0), (1, 0, 0),
              (0, 0, -1), (0, -1, 0), (-1, 0, 0))


def qmul(a: Q, b: Q) -> Q:
    w, x, y, z = a
    v, i, j, k = b
    return (w*v - x*i - y*j - z*k, w*i + x*v + y*k - z*j,
            w*j - x*k + y*v + z*i, w*k + x*j - y*i + z*v)


def qconj(q: Q) -> Q:
    return (q[0], -q[1], -q[2], -q[3])


def qnorm(q: Q) -> Q:
    if len(q) != 4 or any(not math.isfinite(x) for x in q):
        raise ValueError("nonfinite quaternion")
    norm = math.hypot(*q)
    if not math.isfinite(norm) or norm < 1e-12:
        raise ValueError("unrepresentable quaternion")
    return tuple(x/norm for x in q)


def qerror(a: Q, b: Q, transport: Q = ONE) -> float:
    delta = qmul(transport, a)
    return math.fsum((x-y)**2 for x,y in zip(b,delta))


def qblend(a: Q, b: Q, strength: float = .25) -> Q:
    return qnorm(tuple((1-strength)*x + strength*y for x,y in zip(a,b)))


@dataclass(frozen=True)
class Branch:
    ident: int
    parent: int  # -1 is the soma; other values are earlier branch IDs
    voxel: tuple[int, int, int]
    cue: tuple[int, int]
    fast_same: int = 0
    fast_changed: int = 0
    long_same: int = 0
    long_changed: int = 0
    eligibility: int = 0
    drive: int = 0   # short-range axonal excitation / inhibition
    phase: Q = ONE
    forward: Q = ONE  # parent -> this local chart
    observer_frame: Q = ONE

    def prediction(self) -> float:
        same = self.fast_same + self.long_same
        changed = self.fast_changed + self.long_changed
        return (changed + 1) / (same + changed + 2)


@dataclass(frozen=True)
class Cell:
    ident: int
    context: int
    question: int
    soma: tuple[int, int, int]
    atp: int
    substrate: int
    material: int
    branches: tuple[Branch, ...] = ()
    root_phase: Q = ONE
    root_frame: Q = ONE


@dataclass(frozen=True)
class Territory:
    ident: int
    members: tuple[int, ...]
    support: int
    substrate: int = 0
    calcium: int = 0


@dataclass(frozen=True)
class Synapse:
    source: int
    target: int
    branch: int
    kind: str  # excite or inhibit
    delay: int = 1


@dataclass(frozen=True)
class Packet:
    source: int
    target: int
    branch: int
    amount: int
    arrival: int


@dataclass(frozen=True)
class Remainder:
    kind: str
    cell: int
    branch: int
    before: tuple
    reason: str


@dataclass(frozen=True)
class AHState:
    cells: tuple[Cell, ...]
    glia: tuple[Territory, ...]
    synapses: tuple[Synapse, ...] = ()
    pending: tuple[Packet, ...] = ()
    energy_in: int = 0
    material_in: int = 0
    waste: int = 0
    tick: int = 0
    slow_tick: int = 0
    remainders: tuple[Remainder, ...] = ()


class CompartmentalNeuriteEcology:
    """Small traceable 3D organ; no neuron count is a cognition metric.

    Main state is immutable; each accepted operator produces a new state.
    Fast and slow clocks are different. Glia controls slow eligibility only.
    Ordinary real coordinates implement quaternion arithmetic faithfully.
    """
    def __init__(self, *, cells: int = 16, contexts: int = 4,
                 questions: int = 4, branch_cap: int = 4,
                 territory_size: int = 4, side: int = 20,
                 max_synapses: int = 512, max_remainders: int = 4096):
        if not all(type(v) is int for v in (cells, contexts, questions, branch_cap,
                                            territory_size, side, max_synapses, max_remainders)):
            raise ValueError("capacities must be integers")
        if not (1 <= cells <= 256 and 1 <= contexts <= 16 and
                1 <= questions <= 16 and 1 <= branch_cap <= 4 and
                1 <= territory_size <= cells and 8 <= side <= 128 and
                max_synapses >= 0 and max_remainders >= 0):
            raise ValueError("invalid capacity")
        if cells != contexts * questions:
            raise ValueError("one experimentally observable cell per (context,question)")
        columns = math.isqrt(cells-1)+1
        if 2 + 4 * columns >= side:
            raise ValueError("side too small for nonoverlapping somas")
        self.n = cells
        self.contexts = contexts
        self.questions = questions
        self.branch_cap = branch_cap
        self.territory_size = territory_size
        self.side = side
        self.max_synapses = max_synapses
        self.max_remainders = max_remainders
        self.columns = columns
        self.initial_energy = cells * 32 + math.ceil(cells/territory_size) * 4
        self.initial_material = cells * (branch_cap + 4)

    def seed(self) -> AHState:
        cells = tuple(Cell(i, i // self.questions, i % self.questions,
                    (2 + 4 * (i % self.columns),
                     2 + 4 * (i // self.columns), 2),
                    32, 0, self.branch_cap + 4)
                      for i in range(self.n))
        glia = tuple(Territory(k, tuple(range(k*self.territory_size,
                    min(self.n,(k+1)*self.territory_size))), 4)
                     for k in range(math.ceil(self.n/self.territory_size)))
        s = AHState(cells, glia)
        assert self.valid(s)
        return s

    def _replace_cell(self, s: AHState, i: int, value: Cell, **other) -> AHState:
        cells = list(s.cells)
        cells[i] = value
        result = replace(s, cells=tuple(cells), **other)
        if not self.valid(result):
            raise AssertionError("invalid state or broken ledger")
        return result

    def _replace_glia(self, s: AHState, i: int, value: Territory,
                      **other) -> AHState:
        organs = list(s.glia)
        organs[i] = value
        result = replace(s, glia=tuple(organs), **other)
        if not self.valid(result):
            raise AssertionError("invalid glial transition")
        return result

    def cell_id(self, context: int, question: int) -> int:
        if type(context) is not int or type(question) is not int or not (
            0 <= context < self.contexts and 0 <= question < self.questions):
            raise ValueError("invalid context or question")
        return context * self.questions + question

    def territory(self, i: int) -> int:
        return i // self.territory_size

    def ledger(self, s: AHState) -> tuple[int, int]:
        energy = (sum(c.atp + c.substrate for c in s.cells) +
                  sum(g.support + g.substrate for g in s.glia) + s.waste -
                  self.initial_energy - s.energy_in)
        material = (sum(c.material + len(c.branches) for c in s.cells) +
                    len(s.synapses) - self.initial_material - s.material_in)
        return energy, material

    def valid(self, s: AHState) -> bool:
        if not isinstance(s, AHState) or len(s.cells) != self.n or len(s.glia) != math.ceil(self.n/self.territory_size):
            return False
        if any(type(v) is not int or v < 0 for v in (
            s.energy_in, s.material_in, s.waste, s.tick, s.slow_tick)):
            return False
        if len(s.synapses) > self.max_synapses or len(s.remainders) > self.max_remainders:
            return False
        for c in s.cells:
            if (type(c.context) is not int or type(c.question) is not int or
                    not 0 <= c.context < self.contexts or
                    not 0 <= c.question < self.questions or
                    c.ident != self.cell_id(c.context, c.question)):
                return False
            if len(c.soma) != 3 or any(not (type(x) is int and 0 <= x < self.side) for x in c.soma):
                return False
            if any(type(v) is not int or v < 0 for v in (c.atp,c.substrate,c.material)):
                return False
            if len(c.branches) > self.branch_cap or len({b.cue for b in c.branches}) != len(c.branches):
                return False
            positions = {c.soma}
            for idx,b in enumerate(c.branches):
                if b.ident != idx or not (b.parent == -1 or 0 <= b.parent < b.ident):
                    return False
                if not self._valid_cue(b.cue):
                    return False
                if len(b.voxel) != 3 or b.voxel in positions or any(type(x) is not int or not 0 <= x < self.side for x in b.voxel):
                    return False
                parent = c.soma if b.parent == -1 else c.branches[b.parent].voxel
                if sum(abs(x-y) for x,y in zip(parent,b.voxel)) != 1:
                    return False
                positions.add(b.voxel)
                if not all(type(v) is int and 0 <= v <= CAP for v in (
                    b.fast_same,b.fast_changed,b.long_same,b.long_changed)):
                    return False
                if not (type(b.eligibility) is int and 0 <= b.eligibility <= CAP and
                        type(b.drive) is int and -2 <= b.drive <= 2):
                    return False
                for q in (b.phase, b.forward, b.observer_frame):
                    if len(q) != 4 or not all(math.isfinite(x) for x in q) or abs(math.hypot(*q)-1) > 1e-8:
                        return False
            if abs(math.hypot(*c.root_phase)-1) > 1e-8 or abs(math.hypot(*c.root_frame)-1) > 1e-8:
                return False
        for k,g in enumerate(s.glia):
            if g.ident != k or g.members != tuple(range(k*self.territory_size,min(self.n,(k+1)*self.territory_size))):
                return False
            if not all(type(v) is int and v >= 0 for v in (g.support,g.substrate,g.calcium)) or g.calcium > 2:
                return False
        for link in s.synapses:
            if link.kind not in ("excite", "inhibit") or type(link.delay) is not int or not 1 <= link.delay <= 4:
                return False
            if not (0 <= link.source < self.n and 0 <= link.target < self.n and link.source != link.target and
                    0 <= link.branch < len(s.cells[link.target].branches)):
                return False
        for packet in s.pending:
            if (type(packet.arrival) is not int or packet.arrival <= s.tick or
                type(packet.amount) is not int or packet.amount not in (-1,1) or not (
                0 <= packet.source < self.n and 0 <= packet.target < self.n and
                0 <= packet.branch < len(s.cells[packet.target].branches))):
                return False
        return self.ledger(s) == (0,0)

    def _place(self, s: AHState, i: int, parent: int, hint: int) -> tuple[int,int,int]:
        c = s.cells[i]
        old = c.soma if parent == -1 else c.branches[parent].voxel
        occupied = {n.soma for n in s.cells}
        occupied.update(b.voxel for n in s.cells for b in n.branches)
        for j in range(6):
            d = DIRECTIONS[(hint+j) % 6]
            pos = tuple(x+y for x,y in zip(old,d))
            if all(0 <= x < self.side for x in pos) and pos not in occupied:
                return pos
        raise ValueError("no admissible voxel next to selected parent")

    @staticmethod
    def _valid_cue(cue):
        return (isinstance(cue,tuple) and len(cue)==2 and
                all(type(bit) is int and bit in (0,1) for bit in cue))

    def grow(self, s: AHState, context: int, question: int,
             cue: tuple[int,int], parent: int | None = None) -> AHState:
        i = self.cell_id(context,question)
        c = s.cells[i]
        if not self._valid_cue(cue):
            raise ValueError("cue must be two bits")
        if any(b.cue == cue for b in c.branches):
            return s
        if len(c.branches) >= self.branch_cap or c.atp < 1 or c.material < 1:
            raise ValueError("branch/resource capacity exhausted")
        idx = len(c.branches)
        if parent is None:
            parent = -1 if idx == 0 else (idx-1)//2
        if parent != -1 and not (0 <= parent < idx):
            raise ValueError("parent must be an older branch of same cell")
        voxel = self._place(s,i,parent,idx*2 + cue[0]*2 + cue[1])
        parent_phase = c.root_phase if parent == -1 else c.branches[parent].phase
        parent_frame = c.root_frame if parent == -1 else c.branches[parent].observer_frame
        child = Branch(idx,parent,voxel,cue,phase=parent_phase,observer_frame=parent_frame)
        modified = replace(c,atp=c.atp-1,material=c.material-1,
                           branches=c.branches+(child,))
        return self._replace_cell(s,i,modified,waste=s.waste+1)

    @staticmethod
    def plateau(branch: Branch, cue: tuple[int,int]) -> bool:
        # Nonlinear two-cue conjunction, inhibited by a locally delivered axonal packet.
        matches = int(branch.cue[0] == cue[0]) + int(branch.cue[1] == cue[1])
        return branch.drive >= 0 and matches + int(branch.drive > 0) >= 2

    def choose_branch(self, c: Cell, cue: tuple[int,int]) -> Branch | None:
        candidates = [b for b in c.branches if self.plateau(b,cue)]
        if not candidates:
            return None
        # Prefer exact coincidence, then evidence, then deterministic ID.
        return max(candidates,key=lambda b:(int(b.cue == cue),
                   b.fast_same+b.fast_changed+b.long_same+b.long_changed,-b.ident))

    def predict(self, s: AHState, context: int, question: int,
                cue: tuple[int,int]) -> float:
        if not self._valid_cue(cue):
            raise ValueError("cue must be two bits")
        c = s.cells[self.cell_id(context,question)]
        branch = self.choose_branch(c,cue)
        return branch.prediction() if branch is not None else .5

    def observe(self, s: AHState, context: int, question: int,
                cue: tuple[int,int], changed: bool) -> AHState:
        if type(changed) is not bool or not self._valid_cue(cue):
            raise ValueError("invalid cue or boolean witnessed event")
        i = self.cell_id(context,question)
        c = s.cells[i]
        b = self.choose_branch(c,cue)
        if b is None:
            raise ValueError("no active branch; no silent growth upon evidence")
        if c.atp < 1:
            raise ValueError("observation must be ATP-paid")
        # The observation adapter is an explicit A bridge, not a Mat2(R)<->H algebra map.
        y = qnorm((1.,float(cue[0]),float(cue[1]),float(changed)))
        target = qmul(b.observer_frame,y)
        phase = qblend(b.phase,target)
        b2 = replace(b,
                     fast_same=min(CAP,b.fast_same+int(not changed)),
                     fast_changed=min(CAP,b.fast_changed+int(changed)),
                     eligibility=min(CAP,b.eligibility+1),phase=phase)
        branches = list(c.branches)
        branches[b.ident] = b2
        records = [Remainder("[R]_phase_prior",i,b.ident,(b.phase,),
                    "paid local learning changed its previous quaternion state")]
        at_cap = (b.fast_changed == CAP if changed else b.fast_same == CAP)
        if at_cap:
            records.append(Remainder("[R]_witness_cap",i,b.ident,
                   (context,question,cue,changed),
                   "evidence arrived after the local fast counter saturated"))
        if len(s.remainders)+len(records) > self.max_remainders:
            raise ValueError("cannot update without provenance capacity")
        newer = replace(c,atp=c.atp-1,branches=tuple(branches))
        return self._replace_cell(s,i,newer,waste=s.waste+1,
                                  remainders=s.remainders+tuple(records))

    def glial_pulse(self, s: AHState, territory: int) -> AHState:
        if not 0 <= territory < len(s.glia):
            raise ValueError("invalid territory")
        g = s.glia[territory]
        if g.support == 0:
            raise ValueError("glial pool exhausted")
        return self._replace_glia(s,territory,
            replace(g,support=g.support-1,calcium=min(2,g.calcium+1)),
            waste=s.waste+1)

    def slow_step(self, s: AHState, glia_enabled: bool = True) -> AHState:
        updated = []
        remainders = list(s.remainders)
        for c in s.cells:
            g = s.glia[self.territory(c.ident)]
            branches = []
            for b in c.branches:
                if glia_enabled and g.calcium and b.eligibility:
                    lost_same = max(0,b.long_same+b.fast_same-CAP)
                    lost_changed = max(0,b.long_changed+b.fast_changed-CAP)
                    if lost_same or lost_changed:
                        remainders.append(Remainder("[R]_consolidation_cap",
                            c.ident,b.ident,(lost_same,lost_changed),
                            "consolidation overflow; no false memory equivalence"))
                    b = replace(b,
                      long_same=min(CAP,b.long_same+b.fast_same),
                      long_changed=min(CAP,b.long_changed+b.fast_changed),
                      fast_same=0,fast_changed=0,eligibility=0)
                else:
                    b = replace(b,eligibility=max(0,b.eligibility-1))
                branches.append(b)
            updated.append(replace(c,branches=tuple(branches)))
        glia = tuple(replace(g,calcium=max(0,g.calcium-1)) for g in s.glia)
        if len(remainders)>self.max_remainders:
            raise ValueError("cannot consolidate without provenance capacity")
        state = replace(s,cells=tuple(updated),glia=glia,slow_tick=s.slow_tick+1,
                        remainders=tuple(remainders))
        if not self.valid(state):
            raise ValueError("invalid slow transition")
        return state

    def connect(self,s:AHState,source:int,target:int,branch:int,
                kind:str="excite",delay:int=1) -> AHState:
        if kind not in ("excite","inhibit") or type(delay) is not int or not 1<=delay<=4:
            raise ValueError("invalid synaptic type or delay")
        if not(0<=source<self.n and 0<=target<self.n and source!=target and
               0<=branch<len(s.cells[target].branches)):
            raise ValueError("invalid circuit endpoint")
        link=Synapse(source,target,branch,kind,delay)
        if link in s.synapses:
            return s
        c=s.cells[source]
        if c.material<1 or c.atp<1 or len(s.synapses)>=self.max_synapses:
            raise ValueError("contact has no free budget")
        new=replace(c,material=c.material-1,atp=c.atp-1)
        return self._replace_cell(s,source,new,
                  synapses=s.synapses+(link,),waste=s.waste+1)

    def emit(self,s:AHState,source:int) -> AHState:
        if not 0<=source<self.n or s.cells[source].atp<1:
            raise ValueError("spiking requires source ATP")
        links=tuple(e for e in s.synapses if e.source==source)
        if not links:
            raise ValueError("no axonal contacts")
        packets=tuple(Packet(source,e.target,e.branch,
                  1 if e.kind=="excite" else -1,s.tick+e.delay)
                  for e in links)
        return self._replace_cell(s,source,
                replace(s.cells[source],atp=s.cells[source].atp-1),
                pending=s.pending+packets,waste=s.waste+1)

    def fast_step(self,s:AHState) -> AHState:
        now=s.tick+1
        cells=[]
        drive={(packet.target,packet.branch):0 for packet in s.pending
               if packet.arrival==now}
        for packet in s.pending:
            if packet.arrival==now:
                key=(packet.target,packet.branch)
                drive[key]+=packet.amount
        for c in s.cells:
            bs=tuple(replace(b,drive=max(-2,min(2,drive.get((c.ident,b.ident),0))))
                     for b in c.branches)
            cells.append(replace(c,branches=bs))
        result=replace(s,cells=tuple(cells),
            pending=tuple(p for p in s.pending if p.arrival>now),tick=now)
        if not self.valid(result):
            raise ValueError("invalid fast transition")
        return result

    def prune(self,s:AHState,context:int,question:int,branch:int) -> AHState:
        i=self.cell_id(context,question)
        c=s.cells[i]
        if not 0<=branch<len(c.branches):
            raise ValueError("branch absent")
        if any(b.parent==branch for b in c.branches):
            raise ValueError("only leaf branches can be pruned")
        if any(e.target==i and e.branch==branch for e in s.synapses):
            raise ValueError("withdraw contact before pruning")
        if any(p.target==i and p.branch==branch for p in s.pending):
            raise ValueError("pending packet still addresses branch")
        # Only the newest leaf is removable while IDs stay stable.
        if branch != len(c.branches)-1:
            raise ValueError("pruning nonterminal ID requires explicit remapping")
        if len(s.remainders)>=self.max_remainders:
            raise ValueError("cannot prune without recording a typed memory loss")
        removed=c.branches[-1]
        rem=Remainder("[R]_memory_pruned",i,removed.ident,
                      (removed.cue,removed.fast_same,removed.fast_changed,
                       removed.long_same,removed.long_changed,removed.phase),
                      "paid leaf withdrawn; evidence remains attributed")
        newer=replace(c,branches=c.branches[:-1],material=c.material+1)
        return self._replace_cell(s,i,newer,
                                  remainders=s.remainders+(rem,))

    def reframe(self,s:AHState,i:int,root_gauge:Q,
                gauges:tuple[Q,...]) -> AHState:
        c=s.cells[i]
        if len(gauges)!=len(c.branches):
            raise ValueError("independent unit gauge required per live compartment")
        gs=(qnorm(root_gauge),)+tuple(qnorm(g) for g in gauges)
        bs=[]
        for b in c.branches:
            gauge=gs[b.ident+1]
            parent_gauge=gs[0] if b.parent==-1 else gs[b.parent+1]
            bs.append(replace(b,phase=qmul(gauge,b.phase),
                   forward=qmul(qmul(gauge,b.forward),qconj(parent_gauge)),
                   observer_frame=qmul(gauge,b.observer_frame)))
        new=replace(c,root_phase=qmul(gs[0],c.root_phase),
                    root_frame=qmul(gs[0],c.root_frame),branches=tuple(bs))
        return self._replace_cell(s,i,new)

    def defects(self,s:AHState) -> tuple[float,...]:
        out=[]
        for c in s.cells:
            for b in c.branches:
                parent=c.root_phase if b.parent==-1 else c.branches[b.parent].phase
                out.append(qerror(parent,b.phase,b.forward))
        return tuple(out)

    def diagnostics(self,s:AHState) -> dict:
        return {
            "cells":len(s.cells),"compartments":sum(len(c.branches) for c in s.cells),
            "contacts":len(s.synapses),"pending":len(s.pending),
            "territories":len(s.glia),"energy_residual":self.ledger(s)[0],
            "material_residual":self.ledger(s)[1],"energy_spent":s.waste,
            "slow_clock":s.slow_tick,"fast_clock":s.tick,
            "transport_defect_sum":math.fsum(self.defects(s)),
            "memory_pruned":len(s.remainders),
        }
