"""G0-F finite-volume reaction-diffusion and resource-limited neurite surrogate.

This is a controlled mechanochemical surrogate, not a biological neuron model
or a thermodynamic/port-Hamiltonian system. Spatial positions are periodic or
reflecting; chemical fields independently support periodic or no-flux edges.
"""
import argparse
import json
import math
import random


def lap_neighbors(x, y, side, boundary="periodic"):
    if boundary == "periodic":
        return (((x+1) % side, y), ((x-1) % side, y),
                (x, (y+1) % side), (x, (y-1) % side))
    return ((min(x+1, side-1), y), (max(x-1, 0), y),
            (x, min(y+1, side-1)), (x, max(y-1, 0)))


class ChemicalField:
    def __init__(self, side=16, dt=.1, diffusivity=.7, decay=.025,
                 domain_size=16., boundary="periodic"):
        if (side < 4 or dt <= 0 or diffusivity < 0 or decay < 0
                or domain_size <= 0 or boundary not in ("periodic", "no_flux")):
            raise ValueError("invalid chemical parameters")
        self.dx = domain_size / side
        if diffusivity*dt/(self.dx*self.dx) > .25:
            raise ValueError("explicit diffusion CFL exceeded")
        self.side, self.dt, self.D, self.decay = side, dt, diffusivity, decay
        self.domain_size, self.boundary = domain_size, boundary
        self.c = [[0.]*side for _ in range(side)]
        self.balance_residual = 0.
        self.total_source = 0.
        self.total_decay = 0.

    def mass(self):
        return sum(map(sum, self.c))

    def _index(self, x, y):
        if self.boundary == "periodic":
            x, y = x % self.domain_size, y % self.domain_size
        else:
            x = min(self.domain_size, max(0., x))
            y = min(self.domain_size, max(0., y))
        return (min(self.side-1, int(x/self.dx)),
                min(self.side-1, int(y/self.dx)))

    def update(self, sources=None):
        s = self.side
        sources = sources or {}
        before = self.mass()
        reacted = [[0.]*s for _ in range(s)]
        injected = removed = 0.
        for x in range(s):
            for y in range(s):
                prev = self.c[x][y]
                src = max(0., sources.get((x, y), 0.))
                if self.decay:
                    decay_factor = math.exp(-self.decay*self.dt)
                    new = prev*decay_factor + src*(-math.expm1(-self.decay*self.dt))/self.decay
                else:
                    new = prev+self.dt*src
                injected += self.dt*src
                removed += prev+self.dt*src-new
                reacted[x][y] = new
        a = self.D*self.dt/(self.dx*self.dx)
        nxt = [[0.]*s for _ in range(s)]
        for x in range(s):
            for y in range(s):
                v = reacted[x][y]
                nxt[x][y] = (1-4*a)*v+a*sum(
                    reacted[i][j] for i, j in lap_neighbors(x, y, s, self.boundary))
                if nxt[x][y] < -1e-11:
                    raise ArithmeticError("negative chemical concentration")
        self.c = nxt
        residual = (self.mass()-before)-(injected-removed)
        self.balance_residual = max(self.balance_residual, abs(residual))
        self.total_source += injected
        self.total_decay += removed
        return residual

    def sample(self, x, y):
        i, j = self._index(x, y)
        return self.c[i][j]

    def gradient(self, x, y):
        i, j = self._index(x, y)
        if self.boundary == "periodic":
            ix0, ix1 = (i-1) % self.side, (i+1) % self.side
            iy0, iy1 = (j-1) % self.side, (j+1) % self.side
        else:
            ix0, ix1 = max(0, i-1), min(self.side-1, i+1)
            iy0, iy1 = max(0, j-1), min(self.side-1, j+1)
        sx, sy = (ix1-ix0)*self.dx, (iy1-iy0)*self.dx
        return ((self.c[ix1][j]-self.c[ix0][j])/sx if sx else 0.,
                (self.c[i][iy1]-self.c[i][iy0])/sy if sy else 0.)


class Organism:
    def __init__(self, seed=0, n=24, side=16, chemistry=True,
                 nutrient_supply=.12, structural_supply=.12, dt=.1,
                 boundary="periodic", chemo_sign=1, random_navigation=False):
        if (n < 2 or side < 4 or n > side*side or nutrient_supply < 0
                or structural_supply < 0 or dt <= 0 or chemo_sign not in (-1, 1)
                or boundary not in ("periodic", "no_flux")):
            raise ValueError("invalid organism parameters")
        self.rng = random.Random(seed)
        self.n, self.side, self.chemistry = n, side, chemistry
        self.domain_size, self.dt = 16., dt
        self.boundary, self.chemo_sign = boundary, chemo_sign
        self.random_navigation = random_navigation
        self.attract = ChemicalField(side, dt, domain_size=self.domain_size, boundary=boundary)
        self.repulse = ChemicalField(side, dt, diffusivity=.35,
                                     domain_size=self.domain_size, boundary=boundary)
        dx = self.domain_size/side
        cells = self.rng.sample([(x, y) for x in range(side) for y in range(side)], n)
        self.soma = [((x+.5)*dx, (y+.5)*dx) for x, y in cells]
        self.tips = [list(p) for p in self.soma]
        self.exploratory_tips = [list(p) for p in self.soma]
        self.activity = [0.]*n
        self.cell_age = [0.]*n
        self.maturity = [0.]*n
        self.substrate = [.25]*n
        self.atp = [.12]*n
        self.resource = self.atp
        self.structural_material = [.05]*n
        self.branch_length = [0.]*n
        self.contacts = {}
        self.total_growth = 0.
        self.attempted_growth = 0.
        self.stalled = 0
        self.births = 0
        self.prunes = 0
        self.persistent_edges = 0
        self.edge_lifespans = []
        self.supply = nutrient_supply
        self.structural_supply = structural_supply
        self.maturation_time = 5.
        self.persistent_time = 5.
        self.initial_energy = sum(self.substrate)+sum(self.atp)
        self.initial_material = sum(self.structural_material)
        self.nutrient_input = 0.
        self.conversion_loss = 0.
        self.energy_expenditure = 0.
        self.maintenance_expenditure = 0.
        self.exploration_expenditure = 0.
        self.extension_expenditure = 0.
        self.contact_expenditure = 0.
        self.material_input = 0.

    def _position(self, x, y):
        extent = self.domain_size
        if self.boundary == "periodic":
            return x % extent, y % extent
        return min(extent, max(0., x)), min(extent, max(0., y))

    def _distance(self, a, b):
        dx, dy = abs(a[0]-b[0]), abs(a[1]-b[1])
        if self.boundary == "periodic":
            dx, dy = min(dx, self.domain_size-dx), min(dy, self.domain_size-dy)
        return math.hypot(dx, dy)

    def _energy_residual(self):
        pools = sum(self.substrate)+sum(self.atp)
        return self.initial_energy+self.nutrient_input-pools-self.conversion_loss-self.energy_expenditure

    def _material_residual(self):
        inventory = sum(self.structural_material)+sum(self.branch_length)
        return self.initial_material+self.material_input-inventory

    def step(self):
        dt = self.dt
        stim = [self.rng.random() < .20 for _ in range(self.n)]
        self.activity[:] = [.78*a+.22*float(b) for a, b in zip(self.activity, stim)]
        attract_sources, repulse_sources = {}, {}
        for i, (x, y) in enumerate(self.soma):
            cell = self.attract._index(x, y)
            attract_sources[cell] = attract_sources.get(cell, 0.)+.8*self.activity[i]
            repulse_sources[cell] = repulse_sources.get(cell, 0.)+.15*(1-self.activity[i])
        self.attract.update(attract_sources)
        self.repulse.update(repulse_sources)

        degrees = [0]*self.n
        for i, j in self.contacts:
            degrees[i] += 1
            degrees[j] += 1
        for i in range(self.n):
            self.cell_age[i] += dt
            self.maturity[i] = min(1., self.cell_age[i]/self.maturation_time)
            supplied = self.supply*dt
            self.substrate[i] += supplied
            self.nutrient_input += supplied
            material = self.structural_supply*dt
            self.structural_material[i] += material
            self.material_input += material

            converted = min(self.substrate[i], .08*dt)
            efficiency = .8
            self.substrate[i] -= converted
            self.atp[i] += efficiency*converted
            self.conversion_loss += (1-efficiency)*converted
            maintenance = min(self.atp[i], dt*(.002+.0005*degrees[i]))
            self.atp[i] -= maintenance
            self.energy_expenditure += maintenance
            self.maintenance_expenditure += maintenance

            x, y = self.tips[i]
            gx, gy = self.attract.gradient(x, y)
            rx, ry = self.repulse.gradient(x, y)
            theta = (i % 3-1)*.25
            scale = math.sqrt(dt/.1)
            if self.random_navigation:
                heading = self.rng.uniform(-math.pi, math.pi)
                distance = self.rng.uniform(0., .32)*scale
                dx, dy = distance*math.cos(heading), distance*math.sin(heading)
            else:
                dx, dy = self.rng.uniform(-.32, .32)*scale, self.rng.uniform(-.32, .32)*scale
            if self.chemistry and not self.random_navigation:
                dx += self.chemo_sign*(.6*gx-theta*rx)*dt/.1
                dy += self.chemo_sign*(.6*gy-theta*ry)*dt/.1
            self.attempted_growth += math.hypot(dx, dy)
            self.exploratory_tips[i] = list(self._position(x+dx, y+dy))
            exploration_cost = .003*math.hypot(dx, dy)
            if self.atp[i] >= exploration_cost:
                self.atp[i] -= exploration_cost
                self.energy_expenditure += exploration_cost
                self.exploration_expenditure += exploration_cost
            else:
                self.stalled += 1

            # A noisy exploratory displacement only proposes direction. Anatomy
            # changes only when both ATP and structural material pay for extension.
            if self.rng.random() >= .1*dt/.1 or degrees[i] >= 3:
                continue
            length = min(.12, math.hypot(dx, dy))
            material_cost, atp_cost = length, .15*length
            if (length <= 0 or self.structural_material[i] < material_cost
                    or self.atp[i] < atp_cost):
                self.stalled += 1
                continue
            self.structural_material[i] -= material_cost
            self.branch_length[i] += length
            self.total_growth += length
            self.atp[i] -= atp_cost
            self.energy_expenditure += atp_cost
            self.extension_expenditure += atp_cost
            direction = math.atan2(dy, dx)
            self.tips[i] = list(self._position(
                x+length*math.cos(direction), y+length*math.sin(direction)))

            if self.atp[i] < .01:
                continue
            for j, soma in enumerate(self.soma):
                key = tuple(sorted((i, j)))
                if i == j or degrees[j] >= 3 or key in self.contacts:
                    continue
                if self._distance(self.tips[i], soma) < .45 and self.atp[i] >= .02:
                    cost = .02
                    self.atp[i] -= cost
                    self.energy_expenditure += cost
                    self.contact_expenditure += cost
                    self.contacts[key] = {"weight": .24, "age": 0., "mature": False}
                    degrees[i] += 1
                    degrees[j] += 1
                    self.births += 1
                    break

        for pair, edge in list(self.contacts.items()):
            i, j = pair
            age = edge["age"]+dt
            weight = min(1., max(0., edge["weight"]+
                                 .6*self.activity[i]*self.activity[j]*dt-.03*dt))
            mature = edge["mature"] or (age >= self.maturation_time and weight >= .2)
            if mature and not edge["mature"] and age >= self.persistent_time:
                self.persistent_edges += 1
            # Immature connections have an age-dependent survival threshold.
            threshold = .08 if mature else max(.08, .24-.025*age)
            if weight < threshold and age > 1.2:
                del self.contacts[pair]
                self.prunes += 1
                self.edge_lifespans.append(age)
            else:
                edge.update(weight=weight, age=age, mature=mature)
        assert self.valid()

    def valid(self):
        degrees = [0]*self.n
        for (i, j), edge in self.contacts.items():
            if not (0 <= i < j < self.n and edge["age"] >= 0
                    and 0 <= edge["weight"] <= 1):
                return False
            degrees[i] += 1
            degrees[j] += 1
        fields_valid = all(math.isfinite(c) and c >= -1e-12
                           for fld in (self.attract, self.repulse)
                           for row in fld.c for c in row)
        positions_valid = all(math.isfinite(x) and 0 <= x <= self.domain_size
                              for points in (self.tips, self.exploratory_tips)
                              for point in points for x in point)
        pools_valid = all(math.isfinite(x) and x >= -1e-12
                          for pool in (self.substrate, self.atp,
                                       self.structural_material, self.branch_length)
                          for x in pool)
        return (max(degrees, default=0) <= 3 and fields_valid and positions_valid
                and pools_valid and abs(self._energy_residual()) < 1e-8
                and abs(self._material_residual()) < 1e-8)

    def summary(self):
        observed = self.edge_lifespans+[edge["age"] for edge in self.contacts.values()]
        bins = {"<1": 0, "1-5": 0, "5-10": 0, "10+": 0}
        for age in observed:
            bins["<1" if age < 1 else "1-5" if age < 5 else "5-10" if age < 10 else "10+"] += 1
        return {
            "contacts": len(self.contacts), "births": self.births, "prunes": self.prunes,
            "growth": round(self.total_growth, 6), "branch_length": round(sum(self.branch_length), 6),
            "stalls": self.stalled, "mean_substrate": round(sum(self.substrate)/self.n, 6),
            "mean_atp": round(sum(self.atp)/self.n, 6),
            "mean_structural_material": round(sum(self.structural_material)/self.n, 6),
            "exploratory_distance": round(self.attempted_growth, 6),
            "maintenance_cost": round(self.maintenance_expenditure, 6),
            "exploration_cost": round(self.exploration_expenditure, 6),
            "extension_cost": round(self.extension_expenditure, 6),
            "contact_cost": round(self.contact_expenditure, 6),
            "mature_contacts": sum(edge["mature"] for edge in self.contacts.values()),
            "persistent_edge_fraction": round(self.persistent_edges/self.births, 6) if self.births else 0.,
            "edge_lifespan_distribution": bins, "valid": self.valid(),
            "edge_lifespans_right_censored": len(self.contacts),
            "budget_residual_tolerance": 1e-8,
            "max_chemical_balance_error": max(self.attract.balance_residual,
                                               self.repulse.balance_residual),
            "energy_budget_residual": self._energy_residual(),
            "material_budget_residual": self._material_residual(),
        }


def run(seed=0, steps=300, chemistry=True, *, dt=.1, side=16,
        nutrient_supply=.12, structural_supply=.12, chemo_sign=1,
        random_navigation=False, boundary="periodic", n=24):
    if steps < 0:
        raise ValueError("steps must be nonnegative")
    organism = Organism(seed=seed, n=n, side=side, chemistry=chemistry, dt=dt,
                        nutrient_supply=nutrient_supply,
                        structural_supply=structural_supply, chemo_sign=chemo_sign,
                        random_navigation=random_navigation, boundary=boundary)
    for _ in range(steps):
        organism.step()
    return {"seed": seed, "steps": steps, "physical_time": steps*dt,
            "chemistry": chemistry, "chemo_sign": chemo_sign,
            "random_navigation": random_navigation, "boundary": boundary,
            "grid_side": side, "dt": dt, **organism.summary()}


def control_suite(seeds=(0, 1, 2, 3), steps=300, dt=.1, **kwargs):
    """Paired-seed mechanism controls; all conditions use the same seed list."""
    conditions = (
        ("chemistry_on", {"chemistry": True}),
        ("chemistry_off", {"chemistry": False}),
        ("chemotaxis_reversed", {"chemistry": True, "chemo_sign": -1}),
        ("nutrient_ablation", {"nutrient_supply": 0.}),
        ("random_navigation", {"chemistry": False, "random_navigation": True}),
    )
    return [{"condition": name, **run(seed, steps, dt=dt, **kwargs, **options)}
            for seed in seeds for name, options in conditions]


def continuum_suite(seed=0, physical_time=10., dt=.1, side=16, **kwargs):
    """Compare dt/2, 2x spatial resolution, and boundary conditions at fixed time."""
    if physical_time < 0 or dt <= 0 or side < 4:
        raise ValueError("invalid continuum study parameters")
    fine_dt = dt/4
    if not math.isclose(physical_time/fine_dt,
                        round(physical_time/fine_dt), rel_tol=0., abs_tol=1e-9):
        raise ValueError("physical_time must be an integer multiple of dt/4")
    cases = (
        ("baseline", dt, side, "periodic"),
        ("half_timestep", dt/2, side, "periodic"),
        ("refined_grid", fine_dt, side*2, "periodic"),
        ("no_flux_boundary", dt, side, "no_flux"),
    )
    rows = []
    for name, step_dt, grid, boundary in cases:
        steps = round(physical_time/step_dt)
        rows.append({"case": name, **run(seed, steps, dt=step_dt, side=grid,
                                         boundary=boundary, **kwargs)})
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--seeds", default="0,1,2,3")
    parser.add_argument("--dt", type=float, default=.1)
    parser.add_argument("--grid", type=int, default=16)
    parser.add_argument("--suite", action="store_true")
    parser.add_argument("--continuum", action="store_true")
    args = parser.parse_args()
    seeds = [int(value) for value in args.seeds.split(",")]
    if not 1 <= args.steps <= 5000 or not 1 <= len(seeds) <= 32 or args.dt <= 0:
        parser.error("steps must be 1..5000, seeds 1..32, and dt positive")
    if args.suite and args.continuum:
        parser.error("--suite and --continuum are separate experiment modes")
    if args.suite:
        results = control_suite(seeds, args.steps, dt=args.dt, side=args.grid)
    elif args.continuum:
        results = continuum_suite(seeds[0], args.steps*args.dt, args.dt, args.grid)
    else:
        results = [run(seed, args.steps, chemistry=chemistry, dt=args.dt, side=args.grid)
                   for seed in seeds for chemistry in (False, True)]
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
