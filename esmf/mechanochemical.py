"""G0-F: finite-volume inspired reaction-diffusion and resource-limited tips.

A controlled mechanochemical *surrogate*, NOT a biological neuron model.
Periodic boundaries for chemistry and physical positions; no learning labels.
"""
import argparse
import json
import math
import random


def lap_neighbors(x, y, side):
    return (((x+1)%side,y),((x-1)%side,y),(x,(y+1)%side),(x,(y-1)%side))


class ChemicalField:
    def __init__(self, side=16, dt=.1, diffusivity=.7, decay=.025):
        if side<4 or dt<=0 or diffusivity<0 or decay<0:
            raise ValueError("invalid chemical parameters")
        if diffusivity*dt>.25:
            raise ValueError("explicit diffusion CFL exceeded")
        self.side,self.dt,self.D,self.decay=side,dt,diffusivity,decay
        self.c=[[0.]*side for _ in range(side)]
        self.balance_residual=0.
        self.total_source=0.
        self.total_decay=0.

    def mass(self):
        return sum(map(sum,self.c))

    def update(self, sources=None):
        s=self.side
        sources=sources or {}
        before=self.mass()
        # Reaction: exact for first-order decay with constant source per step.
        reacted=[[0.]*s for _ in range(s)]
        injected=removed=0.
        for x in range(s):
            for y in range(s):
                prev=self.c[x][y]
                src=max(0.,sources.get((x,y),0.))
                if self.decay:
                    decay_factor=math.exp(-self.decay*self.dt)
                    new=prev*decay_factor+src*(-math.expm1(-self.decay*self.dt))/self.decay
                else:
                    new=prev+self.dt*src
                injected+=self.dt*src
                removed+=prev+self.dt*src-new
                reacted[x][y]=new
        a=self.D*self.dt
        nxt=[[0.]*s for _ in range(s)]
        for x in range(s):
            for y in range(s):
                v=reacted[x][y]
                nxt[x][y]=(1-4*a)*v+a*sum(reacted[i][j] for i,j in lap_neighbors(x,y,s))
                if nxt[x][y]<-1e-11:
                    raise ArithmeticError("negative chemical concentration")
        self.c=nxt
        # 'injected' tracks input integral, 'removed' tracks source minus
        # exact reacted mass; this is algebraically the true loss plus correction.
        residual=(self.mass()-before)-(injected-removed)
        self.balance_residual=max(self.balance_residual,abs(residual))
        self.total_source+=injected
        self.total_decay+=removed
        return residual

    def sample(self,x,y):
        return self.c[round(x)%self.side][round(y)%self.side]

    def gradient(self,x,y):
        i,j=round(x)%self.side,round(y)%self.side
        return ((self.c[(i+1)%self.side][j]-self.c[(i-1)%self.side][j])/2,
                (self.c[i][(j+1)%self.side]-self.c[i][(j-1)%self.side])/2)


class Organism:
    def __init__(self,seed=0,n=24,side=16,chemistry=True,nutrient_supply=.007):
        self.rng=random.Random(seed)
        self.n,self.side,self.chemistry=n,side,chemistry
        self.attract=ChemicalField(side=side)
        self.repulse=ChemicalField(side=side,diffusivity=.35)
        self.soma=self.rng.sample([(x,y) for x in range(side) for y in range(side)],n)
        self.tips=[list(map(float,p)) for p in self.soma]
        self.activity=[0.]*n
        self.resource=[.25]*n
        self.contacts={}
        self.total_growth=0.
        self.attempted_growth=0.
        self.stalled=0
        self.births=0
        self.prunes=0
        self.supply=nutrient_supply

    def step(self):
        s=self.side
        stim=[self.rng.random()<.20 for _ in range(self.n)]
        self.activity=[.78*a+.22*float(b) for a,b in zip(self.activity,stim)]
        attract_sources={}
        repulse_sources={}
        for i,(x,y) in enumerate(self.soma):
            attract_sources[(x,y)]=attract_sources.get((x,y),0.)+.8*self.activity[i]
            repulse_sources[(x,y)]=repulse_sources.get((x,y),0.)+.15*(1-self.activity[i])
        self.attract.update(attract_sources)
        self.repulse.update(repulse_sources)
        degrees=[0]*self.n
        for i,j in self.contacts:
            degrees[i]+=1;degrees[j]+=1
        for i,(x,y) in enumerate(self.tips):
            self.resource[i]=min(1.,self.resource[i]+self.supply-.0015*degrees[i])
            ga=self.attract.gradient(x,y)
            gr=self.repulse.gradient(x,y)
            theta=(i%3-1)*.25  # heterogeneous local receptor genotype, not target identity
            dx=self.rng.uniform(-.32,.32)
            dy=self.rng.uniform(-.32,.32)
            if self.chemistry:
                dx+=.6*ga[0]-theta*gr[0]
                dy+=.6*ga[1]-theta*gr[1]
            dx=max(-.35,min(.35,dx))
            dy=max(-.35,min(.35,dy))
            travel=math.hypot(dx,dy)
            self.attempted_growth+=travel
            cost=.07*travel
            if cost>self.resource[i]:
                self.stalled+=1
                continue
            self.resource[i]-=cost
            self.total_growth+=travel
            self.tips[i]=[(x+dx)%s,(y+dy)%s]
            if degrees[i]>=3 or self.resource[i]<.05:
                continue
            for j,(a,b) in enumerate(self.soma):
                if i==j or degrees[j]>=3 or tuple(sorted((i,j))) in self.contacts:
                    continue
                ddx=min(abs(self.tips[i][0]-a),s-abs(self.tips[i][0]-a))
                ddy=min(abs(self.tips[i][1]-b),s-abs(self.tips[i][1]-b))
                if ddx*ddx+ddy*ddy<.7**2 and self.resource[i]>=.06:
                    self.resource[i]-=.06
                    self.contacts[tuple(sorted((i,j)))]=[.24,0]
                    degrees[i]+=1;degrees[j]+=1
                    self.births+=1
                    break
        for pair,(w,age) in list(self.contacts.items()):
            i,j=pair
            nw=min(1.,max(0.,w+.06*self.activity[i]*self.activity[j]-.003))
            if nw<.08 and age>12:
                del self.contacts[pair];self.prunes+=1
            else:
                self.contacts[pair]=[nw,age+1]
        assert self.valid()

    def valid(self):
        degrees=[0]*self.n
        for i,j in self.contacts:
            degrees[i]+=1;degrees[j]+=1
        return (max(degrees,default=0)<=3
                and all(0<=r<=1 for r in self.resource)
                and all(math.isfinite(c) and c>=-1e-12 for fld in
                        (self.attract,self.repulse) for row in fld.c for c in row)
                and all(math.isfinite(x) and 0<=x<self.side for tip in self.tips for x in tip))

    def summary(self):
        return {"contacts":len(self.contacts),"births":self.births,"prunes":self.prunes,
                "growth":round(self.total_growth,4),"stalls":self.stalled,
                "mean_resource":round(sum(self.resource)/self.n,4),
                "max_chemical_balance_error":max(self.attract.balance_residual,
                                                 self.repulse.balance_residual)}


def run(seed=0,steps=300,chemistry=True):
    o=Organism(seed=seed,chemistry=chemistry)
    for _ in range(steps):
        o.step()
    return {"seed":seed,"chemistry":chemistry,**o.summary(),"valid":o.valid()}


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--steps",type=int,default=300)
    parser.add_argument("--seeds",default="0,1,2,3")
    args=parser.parse_args()
    if not 1<=args.steps<=5000:
        parser.error("steps between 1 and 5000")
    print(json.dumps([run(s,args.steps,c) for s in map(int,args.seeds.split(","))
                      for c in (False,True)],indent=2))
