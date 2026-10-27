"""G0-D developmental *surrogate*, not a biological brain simulation.

All positions, trophic signals, activity and contact decisions are local.
A 2-D periodic diffusion field is approximated by local neighbor averaging.
Growing tips sample local gradients; contacts stabilize through correlated
activity, mature, consume resources and may later prune. No latent task labels
are consulted by the developmental dynamics.

Standard library only; bounded, seeded, CPU-only.
"""
import argparse
import json
import random


def clamp(x, lo=0., hi=1.):
    return max(lo, min(hi, x))


class Development:
    def __init__(self, seed=0, n=24, side=12, max_degree=4):
        if not 2 <= n <= side*side or side < 3 or max_degree < 1:
            raise ValueError("invalid size")
        self.rng = random.Random(seed)
        self.n, self.side, self.max_degree = n, side, max_degree
        coords = self.rng.sample([(x,y) for x in range(side) for y in range(side)], n)
        self.soma = coords
        self.tips = [[float(x), float(y)] for x,y in coords]
        self.activity = [0.] * n
        self.maturity = [0.] * n
        self.energy = [1.] * n
        self.field = [[0.] * side for _ in range(side)]
        self.contacts = {}  # (i,j) -> [strength, age]
        self.births = self.prunes = 0

    def neighborhood(self, x, y):
        s = self.side
        return [((x+1)%s,y), ((x-1)%s,y), (x,(y+1)%s), (x,(y-1)%s)]

    def step(self, stimulus=None):
        s, n = self.side, self.n
        if stimulus is None:
            stimulus = [self.rng.random() < 0.18 for _ in range(n)]
        if len(stimulus) != n:
            raise ValueError("wrong activity width")
        new_activity = [clamp(0.75*self.activity[i] + 0.25*float(stimulus[i]))
                        for i in range(n)]
        # Local pair interactions; no centralized task objective.
        for (i,j),(w,age) in self.contacts.items():
            new_activity[i] = clamp(new_activity[i]+0.04*w*self.activity[j])
            new_activity[j] = clamp(new_activity[j]+0.04*w*self.activity[i])
        self.activity = new_activity

        # Locally produced trophic signal; four-neighbor finite difference.
        for i,(x,y) in enumerate(self.soma):
            self.field[x][y] += 0.16*self.activity[i]
        nxt = [[0.]*s for _ in range(s)]
        for x in range(s):
            for y in range(s):
                center = self.field[x][y]
                around = sum(self.field[a][b] for a,b in self.neighborhood(x,y))
                nxt[x][y] = clamp(0.91*center+0.08*(around/4-center),0.,3.)
        self.field = nxt

        degree = [0]*n
        for i,j in self.contacts:
            degree[i]+=1;degree[j]+=1

        # Tip navigation: finite-difference chemical gradients, local noise.
        for i,(tx,ty) in enumerate(self.tips):
            self.maturity[i] = clamp(self.maturity[i]+0.002)
            x,y = int(round(tx))%s,int(round(ty))%s
            gx = self.field[(x+1)%s][y]-self.field[(x-1)%s][y]
            gy = self.field[x][(y+1)%s]-self.field[x][(y-1)%s]
            mobility = (1.-0.75*self.maturity[i])*self.energy[i]
            self.tips[i][0] = (tx+mobility*(0.20*gx+self.rng.uniform(-0.45,0.45)))%s
            self.tips[i][1] = (ty+mobility*(0.20*gy+self.rng.uniform(-0.45,0.45)))%s

            if degree[i] >= self.max_degree or self.energy[i]<0.12:
                continue
            for j,(sx,sy) in enumerate(self.soma):
                if j==i or degree[j]>=self.max_degree:
                    continue
                key = tuple(sorted((i,j)))
                if key in self.contacts:
                    continue
                dx = min(abs(self.tips[i][0]-sx),s-abs(self.tips[i][0]-sx))
                dy = min(abs(self.tips[i][1]-sy),s-abs(self.tips[i][1]-sy))
                if dx*dx+dy*dy <= 1.0 and self.rng.random()<0.18:
                    self.contacts[key] = [0.24,0]
                    degree[i]+=1;degree[j]+=1
                    self.energy[i]-=0.06
                    self.births+=1
                    break

        # Synapse stabilization / developmental pruning (surrogate).
        for (i,j),(w,age) in list(self.contacts.items()):
            coincidence = self.activity[i]*self.activity[j]
            w = clamp(w + 0.045*coincidence-0.006)
            if w < 0.07 and age>8:
                del self.contacts[(i,j)]
                self.prunes+=1
            else:
                self.contacts[(i,j)] = [w,age+1]

        for i in range(n):
            incident = sum(1 for (a,b) in self.contacts if i==a or i==b)
            self.energy[i] = clamp(self.energy[i]+0.012-0.002*incident,0.,1.)
        assert self.valid()

    def valid(self):
        deg = [0]*self.n
        for (i,j),(w,age) in self.contacts.items():
            if not (0<=i<j<self.n and 0<=w<=1 and age>=0):
                return False
            deg[i]+=1;deg[j]+=1
        return (max(deg,default=0)<=self.max_degree
                and all(0<=x<=1 for x in self.energy+self.activity+self.maturity)
                and all(0<=t<s for p in self.tips for t,s in zip(p,[self.side]*2))
                and all(0<=v<=3 for line in self.field for v in line))

    def summary(self):
        return {"n":self.n, "contacts":len(self.contacts),
                "births":self.births,"prunes":self.prunes,
                "mean_maturity":round(sum(self.maturity)/self.n,4),
                "mean_resource":round(sum(self.energy)/self.n,4),
                "total_activity":round(sum(self.activity),4)}


def run(seed=0, steps=300):
    obj=Development(seed=seed)
    for _ in range(steps):
        obj.step()
    return {"seed":seed,"steps":steps,**obj.summary(),"valid":obj.valid()}


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--steps",type=int,default=300)
    ap.add_argument("--seeds",default="0,1,2,3")
    args=ap.parse_args()
    seeds=[int(v) for v in args.seeds.split(",")]
    if not 0<=args.steps<=10000 or not 1<=len(seeds)<=32:
        ap.error("steps <=10000 and 1..32 seeds")
    print(json.dumps([run(v,args.steps) for v in seeds],indent=2))


if __name__=="__main__":
    main()
