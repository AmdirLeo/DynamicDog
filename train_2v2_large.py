"""Shared-policy PPO for planar 2v2 soccer.

Four dogs: team A (0,1) attacks +X, team B (2,3) attacks -X.
Each dog sees 23 self/ball features plus teammate and two opponents (6 each).
This is a research simulator; it does not connect to ROS.
"""
import argparse, copy, hashlib, json, math, time
from pathlib import Path
import torch
from torch import nn
from torch.distributions import Normal
import numpy as np

OBS_DIM=41
ACTION_SCALE=torch.tensor((.45,.20,.70))

def export_actor(policy, path):
    """Export the actor without PyTorch for the ROS deployment package."""
    state = policy.state_dict()
    arrays = {}
    for key, value in state.items():
        if key.startswith('actor.') and (key.endswith('.weight') or key.endswith('.bias')):
            layer, field = key.split('.')[1:]
            arrays[f'actor_{layer}_{field}'] = value.detach().cpu().numpy().astype(np.float32)
    arrays.update(obs_dim=np.asarray(OBS_DIM, dtype=np.int64),
                  role=np.asarray('shared_2v2_large'),
                  action_scale=ACTION_SCALE.cpu().numpy().astype(np.float32))
    np.savez(path, **arrays)

class TeamEnv2v2:
    def __init__(self,n,device='cpu',seed=77,horizon=800):
        self.n=n; self.device=torch.device(device); self.horizon=horizon
        self.rng=torch.Generator(device=self.device).manual_seed(seed)
        z=lambda *s:torch.zeros(*s,device=self.device)
        self.p=z(n,4,2);self.v=z(n,4,2);self.yaw=z(n,4);self.omega=z(n,4)
        self.ball=z(n,2);self.bv=z(n,2);self.prev=z(n,4,3);self.touch=z(n,4)
        self.age=z(n);self.dead_steps=torch.zeros(n,dtype=torch.long,device=self.device)
        self.service=torch.tensor([[-2.5,-1.25],[-2.5,0],[-2.5,1.25],[0,-1.25],[0,0],[0,1.25],[2.5,-1.25],[2.5,0],[2.5,1.25]],device=self.device)
        self.last=torch.zeros(n,8,dtype=torch.bool,device=self.device)
        self.last_collision=z(n); self.last_contact=torch.zeros(n,4,dtype=torch.bool,device=self.device)
        self.reset(torch.ones(n,dtype=torch.bool,device=self.device))
    def rand(self,*s):return torch.rand(*s,generator=self.rng,device=self.device)
    def uni(self,a,b):return a+(b-a)*self.rand(self.n)
    def wrap(self,x):return torch.atan2(x.sin(),x.cos())
    def body(self,v,y):return torch.stack((y.cos()*v[...,0]+y.sin()*v[...,1],-y.sin()*v[...,0]+y.cos(v[...,1]) if False else -y.sin()*v[...,0]+y.cos()*v[...,1]),-1)
    def reset(self,m):
        b=torch.stack((self.uni(-2.0,2.0),self.uni(-1.7,1.7)),-1)
        # Team A behind ball, team B between ball and +X goal. Teammates are
        # spread laterally so the first policy can learn support behavior.
        a0=b+torch.stack((self.uni(-1.7,-.6),self.uni(-.55,.55)),-1)
        a1=b+torch.stack((self.uni(-1.7,-.6),self.uni(-1.5,1.5)),-1)
        b0=b+torch.stack((self.uni(.6,1.8),self.uni(-.55,.55)),-1)
        b1=b+torch.stack((self.uni(.6,1.8),self.uni(-1.5,1.5)),-1)
        p=torch.stack((a0,a1,b0,b1),1);p[:,:,1].clamp_(-2.35,2.35)
        yaw=torch.zeros(self.n,4,device=self.device)
        yaw[:,:2]=torch.atan2(b[:,None,1]-p[:,:2,1],b[:,None,0]-p[:,:2,0])
        yaw[:,2:]=torch.atan2(b[:,None,1]-p[:,2:,1],b[:,None,0]-p[:,2:,0])+math.pi
        yaw=self.wrap(yaw+(self.rand(self.n,4)*2-1)*.8)
        self.p[m]=p[m];self.yaw[m]=yaw[m];self.ball[m]=b[m]
        self.bv[m]=(self.rand(self.n,2)*2-1)[m]*.12
        for x in (self.v,self.omega,self.prev,self.touch,self.age,self.dead_steps,self.last_collision):x[m]=0
        self.last_contact[m]=False
    def observations(self):
        out=[]
        for i in range(4):
            team=i//2; sign=1. if team==0 else -1.
            teammate=1-i if i<2 else 5-i
            opp=(2,3) if team==0 else (0,1)
            p=self.p[:,i];y=self.yaw[:,i];rel=self.body(self.ball-p,y)
            goal=torch.stack((torch.full_like(y,sign*4.585),torch.zeros_like(y)),-1)
            core=torch.cat((rel/3,self.body(self.bv,y)/2,self.body(goal-p,y)/5,
                self.body(self.v[:,i],y),self.omega[:,i,None],self.prev[:,i],
                self.wrap(y-(0 if team==0 else math.pi)).sin()[:,None],
                self.wrap(y-(0 if team==0 else math.pi)).cos()[:,None],
                ((sign*self.p[:,i,0])+5)[:,None]/9,((5-sign*self.p[:,i,0])[:,None])/9,
                (self.p[:,i,1]+2.75)[:,None]/6,(2.75-self.p[:,i,1])[:,None]/6,
                torch.linalg.vector_norm(rel,dim=-1)[:,None]/3,self.touch[:,i,None],
                self.age[:,None]/.5,self.age[:,None]/.5,(self.age<=.5).float()[:,None]),-1)
            def other(j):
                return torch.cat((self.body(self.p[:,j]-p,y)/3,
                    self.body(self.v[:,j]-self.v[:,i],y)/2,
                    self.wrap(self.yaw[:,j]-y).sin()[:,None],
                    self.wrap(self.yaw[:,j]-y).cos()[:,None]),-1)
            out.append(torch.cat((core,other(teammate),other(opp[0]),other(opp[1])),-1).clamp(-5,5))
        return torch.stack(out,1)
    def step(self,actions):
        actions=actions.clamp(-1,1);oldb=self.ball.clone();oldp=self.p.clone();old_dist=torch.linalg.vector_norm(self.ball[:,None]-self.p,dim=-1)
        collision=torch.zeros(self.n,dtype=torch.bool,device=self.device);contact=torch.zeros(self.n,4,dtype=torch.bool,device=self.device);impact=torch.zeros(self.n,4,device=self.device)
        for _ in range(2):
            dt=.025;desired=actions*ACTION_SCALE.to(self.device);alpha=torch.full((self.n,4),.25,device=self.device)
            desired_world=torch.stack((self.yaw.cos()*desired[:,:,0]-self.yaw.sin()*desired[:,:,1],self.yaw.sin()*desired[:,:,0]+self.yaw.cos()*desired[:,:,1]),-1)
            self.v+=alpha[:,:,None]*(desired_world-self.v);self.omega+=alpha*(desired[:,:,2]-self.omega);self.p+=self.v*dt;self.yaw=self.wrap(self.yaw+self.omega*dt)
            self.bv*=.97;self.ball+=self.bv*dt
            for i in range(4):
                for j in range(i+1,4):
                    delta=self.p[:,j]-self.p[:,i];dist=torch.linalg.vector_norm(delta,dim=-1).clamp(min=1e-6);norm=delta/dist[:,None];touching=dist<.52;collision|=touching
                    rel=(self.v[:,j]-self.v[:,i])*norm;approach=torch.stack((rel[:,0],-rel[:,0]),-1).abs();impact[:,i]+=approach[:,0].square()*.05*touching;impact[:,j]+=approach[:,1].square()*.05*touching
                    corr=(.52-dist).clamp(min=0)[:,None]*norm;self.p[:,i]-=.5*corr;self.p[:,j]+=.5*corr
            delta=self.ball[:,None]-self.p;dist=torch.linalg.vector_norm(delta,dim=-1).clamp(min=1e-6);norm=delta/dist[:,:,None];touching=dist<.38;contact|=touching
            rel=self.bv[:,None]-self.v;j=(-1.3*(rel*norm).sum(-1)/1.04).clamp(min=0)*touching;tangent=torch.stack((-norm[:,:,1],norm[:,:,0]),-1);jt=(-(rel*tangent).sum(-1)/1.04).clamp(-.15*j,.15*j);change=j[:,:,None]*norm+jt[:,:,None]*tangent;self.bv+=change.sum(1);self.v-=.04*change;self.ball+=.8*((.38-dist).clamp(min=0)[:,:,None]*norm).sum(1)
        def cross(sign):
            before=oldb[:,0]*sign;after=self.ball[:,0]*sign;f=((4.705-before)/(after-before).clamp(min=1e-6)).clamp(0,1);cy=oldb[:,1]+f*(self.ball[:,1]-oldb[:,1]);return (before<4.705)&(after>=4.705)&(cy.abs()<=.63)
        goal=cross(1);counter=cross(-1)&~goal
        out=((self.p[:,:,0].abs()>4.95)|(self.p[:,:,1].abs()>2.70));aout=out[:,:2].any(1);bout=out[:,2:].any(1)
        dead=((self.ball[:,1].abs()>1.95)|((self.ball[:,0].abs()>4.20)&(self.ball[:,1].abs()>.75))|(self.ball[:,0].abs()>5.12));gap=torch.cdist(self.p,self.p).masked_fill(torch.eye(4,device=self.device,dtype=torch.bool)[None],99).amin((1,2));slow=torch.linalg.vector_norm(self.v,dim=-1).max(1).values<.08;stuck=(gap<.56)&slow;self.dead_steps=torch.where(stuck,self.dead_steps+1,torch.zeros_like(self.dead_steps));deadlock=(self.dead_steps>=160)&~(goal|counter|aout|bout);reset=dead|deadlock;timeout=(self.age+1>=self.horizon)&~(goal|counter|aout|bout);done=goal|counter|aout|bout|timeout
        target_a=torch.tensor([4.585,0.],device=self.device);target_b=torch.tensor([-4.585,0.],device=self.device);progress_a=torch.linalg.vector_norm(oldb-target_a,dim=-1)-torch.linalg.vector_norm(self.ball-target_a,dim=-1);progress_b=torch.linalg.vector_norm(oldb-target_b,dim=-1)-torch.linalg.vector_norm(self.ball-target_b,dim=-1);team_a=5*progress_a+20*goal.float()-20*counter.float()-.003-.02*collision.float()-impact[:,:2].sum(1)-10*aout.float();team_b=5*progress_b+20*counter.float()-20*goal.float()-.003-.02*collision.float()-impact[:,2:].sum(1)-10*bout.float();r=torch.stack((team_a,team_a,team_b,team_b),-1)
        self.age+=1;nearest=torch.linalg.vector_norm(self.ball[:,None]-self.service[None],dim=-1).argmin(-1);self.ball=torch.where(reset[:,None],self.service[nearest],self.ball);self.bv=torch.where(reset[:,None],torch.zeros_like(self.bv),self.bv);self.dead_steps=torch.where(reset,torch.zeros_like(self.dead_steps),self.dead_steps);self.last=torch.stack((goal,dead,aout,bout,counter,timeout,deadlock,reset),-1);self.last_collision=collision;self.last_contact=contact;self.reset(done)
        return r,done

class Policy(nn.Module):
    """Six hidden layers of 256 units for both actor and critic."""
    HIDDEN = 256
    DEPTH = 6
    def __init__(self):
        super().__init__()
        def mlp(out_dim):
            layers=[]
            inp=OBS_DIM
            for _ in range(self.DEPTH):
                layers.extend((nn.Linear(inp,self.HIDDEN),nn.Tanh()))
                inp=self.HIDDEN
            layers.append(nn.Linear(self.HIDDEN,out_dim))
            return nn.Sequential(*layers)
        self.actor=mlp(3)
        self.critic=mlp(1)
        self.logstd=nn.Parameter(torch.full((3,),-.6))
    def sample(self,o):
        d=Normal(self.actor(o),self.logstd.clamp(-3,1).exp());raw=d.sample();a=raw.tanh();lp=(d.log_prob(raw)-torch.log(1-a.square()+1e-6)).sum(-1);return a,raw,lp,self.critic(o).squeeze(-1)
    def score(self,o,raw):
        d=Normal(self.actor(o),self.logstd.clamp(-3,1).exp());lp=(d.log_prob(raw)-torch.log(1-raw.tanh().square()+1e-6)).sum(-1);return lp,d.entropy().sum(-1),self.critic(o).squeeze(-1)

def update(m,opt,o,raw,oldlp,adv,target):
    adv=(adv-adv.mean())/(adv.std()+1e-8)
    for epoch in range(4):
        kl=0;num=0
        for ix in torch.randperm(len(o),device=o.device).split(4096):
            lp,ent,val=m.score(o[ix],raw[ix]);ratio=(lp-oldlp[ix]).exp();loss=-torch.minimum(ratio*adv[ix],ratio.clamp(.8,1.2)*adv[ix]).mean()+.5*(val-target[ix]).square().mean()-.004*ent.mean();opt.zero_grad();loss.backward();nn.utils.clip_grad_norm_(m.parameters(),.5);opt.step();kl+=float(((ratio-1)-(lp-oldlp[ix])).mean().detach())*len(ix);num+=len(ix)
        if kl/num>.025:break
    return float(loss.detach()),kl/num

def evaluate(m,device,seed,count=128,horizon=800):
    e=TeamEnv2v2(count,device,seed,horizon);live=torch.ones(count,dtype=torch.bool,device=device);goals=torch.zeros(count,dtype=torch.bool,device=device);counter=torch.zeros_like(goals);timeouts=torch.zeros_like(goals)
    for _ in range(horizon):
        with torch.no_grad():a=m.actor(e.observations()).tanh()
        _,done=e.step(a);goals|=e.last[:,0]&live;counter|=e.last[:,4]&live;timeouts|=e.last[:,5]&live;live&=~done
    return dict(episodes=count,goals=int(goals.sum()),goal_rate=float(goals.float().mean()),counter_goals=int(counter.sum()),timeouts=int(timeouts.sum()))

def main():
    p=argparse.ArgumentParser();p.add_argument('--device',default='cuda:0');p.add_argument('--iterations',type=int,default=360);p.add_argument('--envs',type=int,default=256);p.add_argument('--steps',type=int,default=96);p.add_argument('--seed',type=int,default=77);p.add_argument('--eval-every',type=int,default=60);p.add_argument('--eval-envs',type=int,default=64);p.add_argument('--eval-horizon',type=int,default=400);p.add_argument('--output',default='policy_2v2_shared_large.pt');p.add_argument('--initial',default='');p.add_argument('--smoke',action='store_true');a=p.parse_args();torch.set_num_threads(4);torch.manual_seed(a.seed);dev=torch.device(a.device);m=Policy().to(dev)
    print(json.dumps({'network': {'hidden': Policy.HIDDEN, 'depth': Policy.DEPTH, 'parameters': sum(p.numel() for p in m.parameters())}}),flush=True)
    if a.initial:
        checkpoint = torch.load(a.initial, map_location=dev, weights_only=False)
        state = checkpoint.get('state_dict', checkpoint)
        missing, unexpected = m.load_state_dict(state, strict=False)
        if missing or unexpected:
            raise ValueError(f'初始模型结构不匹配：missing={missing}, unexpected={unexpected}')
        print(json.dumps({'loaded_initial': a.initial}), flush=True)
    opt=torch.optim.Adam(m.parameters(),lr=7e-5,eps=1e-5)
    if a.smoke:
        print(json.dumps({'obs_dim':OBS_DIM,'device':str(dev),'initial':evaluate(m,dev,20261119,32,100)}),flush=True)
        e=TeamEnv2v2(32,dev,a.seed,100);o=e.observations();assert o.shape==(32,4,OBS_DIM),o.shape;act,raw,lp,val=m.sample(o.reshape(-1,OBS_DIM));r,done=e.step(act.reshape(32,4,3));adv=torch.randn_like(val).detach();metrics=update(m,opt,o.reshape(-1,OBS_DIM),raw.detach(),lp.detach(),adv,(adv+val.detach()));print(json.dumps({'smoke_pass':True,'shape':list(o.shape),'reward_mean':float(r.mean()),'update':metrics}),flush=True);return
    initial_eval=evaluate(m,dev,20261119,a.eval_envs,a.eval_horizon)
    best=copy.deepcopy(m.state_dict());best_rate=initial_eval['goal_rate'];best_it=0
    t0=time.time();n=a.envs;T=a.steps
    for it in range(1,a.iterations+1):
        e=TeamEnv2v2(n,dev,a.seed+it,800);ob=torch.empty(T,n,4,OBS_DIM,device=dev);raw=torch.empty(T,n,4,3,device=dev);lp=torch.empty(T,n,4,device=dev);val=torch.empty_like(lp);rew=torch.empty_like(lp);doneb=torch.empty_like(lp)
        with torch.no_grad():
            for s in range(T):
                oo=e.observations();act,rr,ll,v=m.sample(oo.reshape(-1,OBS_DIM));rr=rr.reshape(n,4,3);act=act.reshape(n,4,3);ra,done=e.step(act);ob[s]=oo;raw[s]=rr;lp[s]=ll.reshape(n,4);val[s]=v.reshape(n,4);rew[s]=ra;doneb[s]=done[:,None].expand(-1,4).float()
            boot=m.critic(e.observations().reshape(-1,OBS_DIM)).reshape(n,4).detach();ad=torch.empty_like(rew);last=torch.zeros(n,4,device=dev)
            for s in reversed(range(T)):
                nv=boot if s==T-1 else val[s+1];delta=rew[s]+.997*nv*(1-doneb[s])-val[s];last=delta+.997*.95*(1-doneb[s])*last;ad[s]=last;target=ad+val
        loss,kl=update(m,opt,ob.reshape(-1,OBS_DIM),raw.reshape(-1,3),lp.reshape(-1),ad.reshape(-1),target.reshape(-1))
        if it%10==0:print(json.dumps({'iteration':it,'steps':it*T*n*4,'seconds':round(time.time()-t0,1),'rollout_goals':int(e.last[:,0].sum()),'loss':loss,'kl':kl}),flush=True)
        if it%a.eval_every==0 or it==a.iterations:
            res=evaluate(m,dev,20261120,a.eval_envs,a.eval_horizon);print(json.dumps({'validation':dict(iteration=it,**res)}),flush=True)
            if res['goal_rate']>best_rate:best_rate=res['goal_rate'];best=copy.deepcopy(m.state_dict());best_it=it;print(json.dumps({'selected_iteration':it}),flush=True)
    m.load_state_dict(best);held={};
    for seed in (20261121,20261122):held[seed]=evaluate(m,dev,seed,a.eval_envs*2,a.eval_horizon);print(json.dumps({'heldout':seed,**held[seed]}),flush=True)
    out={'state_dict':{k:v.cpu() for k,v in m.state_dict().items()},'obs_dim':OBS_DIM,'action_scale':tuple(float(x) for x in ACTION_SCALE),'role':'shared_2v2_large','network':{'hidden':Policy.HIDDEN,'depth':Policy.DEPTH},'field':{'outer_length':10.,'width':5.5,'goal_spacing':9.17,'goal_width':1.5,'deadzone_inset':.8,'deadlock_seconds':8.},'best_iteration':best_it,'validation':res,'heldout':held,'training_steps':a.iterations*T*n*4,'args':vars(a)};torch.save(out,a.output)
    actor_path = str(Path(a.output).with_name(Path(a.output).stem + '_actor.npz'))
    export_actor(m, actor_path)
    print(json.dumps({'complete':True,'output':a.output,'actor_output':actor_path,'best_iteration':best_it}),flush=True)
if __name__=='__main__':main()
