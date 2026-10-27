#!/usr/bin/env python3
"""Rebuilt ACT4 page. Physical-mm layout, vector artwork, explicit semantic colors.
Run validate_sources.py first. Only display offsets are spread; measured values
are never jittered. Scalar inputs, curves and bin counts are exported to CSV.
"""
from pathlib import Path
import argparse,json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle,FancyBboxPatch
from matplotlib.colors import LinearSegmentedColormap,LogNorm,to_rgb
from scipy.stats import gaussian_kde
O=Path(__file__).resolve().parent
read=lambda n:pd.read_csv(O/'data'/n,sep='\t')
A,B,C,E,F,G,EX=[read(n) for n in ['A.tsv','B.tsv','C_D.tsv','E.tsv','F.tsv','G_null_draws.tsv','G_examples.tsv']]
for d in [B,C,E,F,G,EX]:d['id']=d.signature_id.str.replace('PF4C-','',regex=False)
Q=[.25,.5,.75,1.];IDS=['B1-004','B4-007','B4-029','FINAL-053','FINAL-054','FINAL-055','FINAL-056','FINAL-079']
INK='#24272E';BLOCK='#303965';NAIVE='#C65C48';BOTH='#8E6B9E';NEUTRAL='#93908C';WARM='#ECEAE6';HAIR='#D7D6D2';TEAL='#3B9293'
QC=['#819CA8','#549992','#BE9850','#B66D85']
REC=LinearSegmentedColormap.from_list('recovery',['#F5F7F0','#D0E6DD','#85BEBD','#347F85','#104F5A'],N=256)
plt.rcParams.update({'font.family':'Arial','font.size':6.7,'text.color':INK,'axes.labelcolor':INK,'axes.edgecolor':'#6C6D70','axes.labelsize':7,'axes.linewidth':.45,'axes.spines.top':False,'axes.spines.right':False,'xtick.color':INK,'ytick.color':INK,'xtick.labelsize':6.2,'ytick.labelsize':6.2,'xtick.major.size':2,'ytick.major.size':2,'xtick.major.width':.45,'ytick.major.width':.45,'lines.linewidth':1.,'svg.fonttype':'none','pdf.fonttype':42,'pdf.compression':6,'savefig.facecolor':'white'})
TH=-np.log10(.05);records=[]
def record(panel,kind,**kw):records.append(dict(panel=panel,kind=kind,**kw))
def density(vals,grid,reflect=False):
 k=gaussian_kde(np.asarray(vals));y=k(grid)
 if reflect:y+=k(-grid)+k(2-grid)
 return y
class Panel:
 def __init__(self,fig,x,y,w,h,pw=210,ph=297):self.fig,self.x,self.y,self.w,self.h,self.pw,self.ph=fig,x,y,w,h,pw,ph
 def ax(self,x,y,w,h):return self.fig.add_axes([(self.x+x)/self.pw,1-(self.y+y+h)/self.ph,w/self.pw,h/self.ph])
 def text(self,x,y,s,size=6.7,color=INK,weight='normal',**kw):return self.fig.text((self.x+x)/self.pw,1-(self.y+y)/self.ph,s,fontsize=size,color=color,weight=weight,va='top',**kw)
 def title(self,l,title):self.text(0,-.5,l,11,weight='bold');self.text(5.3,0,title,8.1,weight='bold',linespacing=1.24)
 def line(self,x1,y1,x2,y2,color=HAIR,lw=.5):
  self.fig.add_artist(plt.Line2D([(self.x+x1)/self.pw,(self.x+x2)/self.pw],[1-(self.y+y1)/self.ph,1-(self.y+y2)/self.ph],transform=self.fig.transFigure,c=color,lw=lw))
 def rect(self,x,y,w,h,color,rounding=0,edge='none',lw=.3):
  a=self.ax(x,y,w,h);a.set_xlim(0,w);a.set_ylim(0,h);a.axis('off');a.add_patch(FancyBboxPatch((0,0),w,h,boxstyle=f'round,pad=0,rounding_size={rounding}',fc=color,ec=edge,lw=lw))
def swarm(y,max_width=.18,step=.042,dy=.048):
 # Stable beeswarm: only x is displaced; y stays exactly equal to the source.
 placed=[];x=np.zeros(len(y))
 for i in np.argsort(y,kind='stable'):
  for off in [0]+[v for j in range(1,30) for v in [j*step,-j*step]]:
   if all(abs(y[i]-yy)>=dy or abs(off-xx)>=step*.99 for xx,yy in placed):break
  x[i]=off;placed.append((off,y[i]))
 if np.max(abs(x))>max_width:x*=max_width/np.max(abs(x))
 return x

def panel_A(p):
 p.title('A','Real DS03 block-structured\nmeasurement landscape')
 p.text(5,10,'B4-007 · 57 malignant regional units · 12 donors',6.2,color=NEUTRAL)
 d=read('A_landscape.tsv');donors=sorted(d.block_id.unique(),key=lambda x:int(x[2:]));ax=p.ax(15,19,62,34)
 for i,donor in enumerate(donors):
  z=d[d.block_id==donor];ax.axhline(i,c=HAIR,lw=.35,ls=':')
  for middle,col in [(0,'#C8C7C2'),(1,TEAL)]:
   zz=z[z.middle_member==middle];ax.scatter(zz.S,np.full(len(zz),i),s=10,c=col,ec='white',lw=.35,zorder=3)
 ax.set_yticks(range(12),[f'{donor} ({sum(d.block_id==donor)})'+(' *' if d[d.block_id==donor].perturbable_q100.iloc[0]=='YES' else '') for donor in donors],fontsize=6)
 ax.set_ylim(11.7,-.7);ax.tick_params(axis='y',length=0,pad=2);ax.set_xlabel('Frozen signature score S',labelpad=1);ax.set_ylabel('Donor (units)',labelpad=1);ax.spines['left'].set_visible(False)
 p.text(5,63,'● Endpoint / source-window',6,color=TEAL);p.text(53,63,'● Other unit',6,color=NEUTRAL)
 p.text(5,68,'* q100 perturbable block (not an individual-unit assignment)',6,color=NEUTRAL)


def panel_B(p):
 p.title('B','Calibrated operating regimes differ across contexts')
 p.text(8,9,'Ivy · 61 signatures',6.8,weight='bold')
 p.text(51,9,'DS03 · 8 historical identities',6.8,weight='bold')
 left=p.ax(8,18,34,34);right=p.ax(51,18,28,34)
 for k,q in enumerate(Q):
  z=B[B.q==q].sort_values('signature_id');v=z.fraction.to_numpy();off=swarm(v,max_width=.30,step=.065,dy=.035)
  left.scatter(k+off,v,s=3.5,c=QC[k],alpha=.65,edgecolors='none')
  lo,med,hi=np.quantile(v,[.25,.5,.75]);left.plot([k,k],[lo,hi],c=INK,lw=1.4);left.scatter(k,med,s=9,c=INK,zorder=4)
  for sid,yy,xx in zip(z.signature_id,v,off):record('B','signature_recovery',signature_id=sid,q=q,recovery=yy,display_x=k+xx)
 colors={'B1-004':'#759696','B4-007':'#104F5A','FINAL-056':'#397E82','B4-029':'#747470','FINAL-079':'#A3A19C'}
 for sid in IDS:
  z=C[C.id==sid].sort_values('q');v=z['sum'].to_numpy()/20
  if sid=='FINAL-053':right.plot(range(4),v,'--o',c='#C8C6C1',mfc='white',mec='#C8C6C1',ms=3,lw=.6,zorder=5)
  else:right.plot(range(4),v,'-o',color=colors.get(sid,'#B6B4AF'),ms=2.3,lw=1 if sid in colors else .55,alpha=.95 if sid in colors else .65,zorder=3)
  for q,yy in zip(Q,v):record('B','historical_identity_recovery',signature_id=sid,q=q,recovery=yy,formal_adjudicable=sid!='FINAL-053')
 for label,yy,col in [('B1-004',1.04,colors['B1-004']),('B4-007',.94,colors['B4-007']),('FINAL-056',.84,colors['FINAL-056']),('B4-029',.10,colors['B4-029']),('FINAL-079',0,colors['FINAL-079'])]:
  target=1 if yy>.5 else 0;right.annotate(label,xy=(3,target),xytext=(3.25,yy),fontsize=6,color=col,va='center',annotation_clip=False,arrowprops=dict(arrowstyle='-',color=col,lw=.4,shrinkA=0,shrinkB=2))
 for ax in [left,right]:
  ax.set_ylim(-.04,1.07);ax.set_xlim(-.4,3.15);ax.set_xticks(range(4),['25','50','75','100']);ax.set_xlabel('q (%)',labelpad=1);ax.set_yticks([0,.5,1]);ax.tick_params(pad=1);ax.spines['bottom'].set_color(HAIR)
 left.set_ylabel('Fraction recovered',labelpad=1);right.set_yticklabels([]);right.tick_params(axis='y',length=0);right.spines['left'].set_visible(False)
 p.text(8,14,'42/61 complete at q100',6,color=QC[3])
 p.text(8,62,'FINAL-053: descriptive only; exact α=.05 unreachable',6,color=NEUTRAL)
 p.text(8,67,'7 formally adjudicable identities in C; historical n=8 here',6,color=NEUTRAL)
 p.text(8,73,'Blocked-null size: Ivy .053 · DS03 .044 (nominal α=.05)',6,color=BLOCK)


def panel_C(p):
 p.title('C','DS03 recovery after exact-inference qualification')
 p.line(5,9,190,9,HAIR,.5)
 p.text(5,12,'8 frozen identities  ·  7 adjudicable at α = .05',7.2,weight='bold')
 p.text(5,18,'○ FINAL-053 gated out: exact p min = .25 (q25/q50); .0625 (q75/q100)',6.5,color=NEUTRAL)
 p.text(5,24,'All 640 positive worlds passed legality and invariance gates',6.2,color=NEUTRAL)
 ids=[i for i in IDS if i!='FINAL-053'];mx=27;cw=30
 for j,col in enumerate(QC):
  p.rect(mx+j*cw,32,cw-1.4,1.3,col);p.text(mx+j*cw+(cw-1.4)/2,35,f'q{int(Q[j]*100)}',7.2,color=col,ha='center',weight='bold')
 p.text(158,33,'K(q100)',6.8,ha='center',weight='bold');p.text(179,33,'q100 outcome',6.8,ha='center',weight='bold')
 for i,sid in enumerate(ids):
  yy=42+i*7.4;p.text(2,yy+1.6,sid,7.7,weight='bold' if sid in ['B4-007','B4-029','FINAL-056'] else 'normal')
  for j,q in enumerate(Q):
   r=C[(C.id==sid)&(C.q==q)].iloc[0];v=r['sum']/20;p.rect(mx+j*cw,yy,cw-1.4,6.5,REC(v),.5,edge='#DEE5DF' if v==0 else 'none');p.text(mx+j*cw+(cw-1.4)/2,yy+1.5,f'{int(r["sum"])}/20',8.3,color='white' if v>.7 else '#24474E',ha='center',weight='bold' if v>.7 else 'normal')
  r=C[(C.id==sid)&(C.q==1)].iloc[0];p.text(158,yy+1.6,str(r.K_target),7.4,ha='center');p.text(179,yy+1.6,f'{int(r["sum"])}/20',7.4,ha='center')
 a=p.ax(27,97,48,2.4);a.axis('off');a.set_xlim(0,1);a.set_ylim(0,1)
 for j in range(100):a.add_patch(Rectangle((j/100,0),.0102,1,fc=REC(j/99),ec='none',antialiased=False))
 for x,t in [(27,'0'),(51,'0.5'),(75,'1')]:p.text(x,101,t,6,ha='center')
 p.text(80,97,'Recovery fraction · 20 worlds per cell',6.3)


def panel_D(p):
 p.title('D','Matched legal capacity\ndoes not determine recovery')
 p.text(5.3,10,'B4-007 / B4-029 · E = 10',6.3,color=NEUTRAL)
 # Shared category positions; space at right reserved for direct endpoint labels.
 ax=p.ax(9,21,31,15);ax.plot(range(4),[2,4,5,7],'-o',c=NEUTRAL,ms=3.4,mec='white',mew=.4)
 for x,k in enumerate([2,4,5,7]):ax.annotate(str(k),(x,k),xytext=(0,4),textcoords='offset points',ha='center',fontsize=6.3)
 ax.set_xlim(-.3,3.3);ax.set_ylim(0,9);ax.set_yticks([0,4,8]);ax.set_ylabel('Legal K',labelpad=1);ax.tick_params(axis='x',bottom=False,labelbottom=False);ax.spines['bottom'].set_visible(False)
 p.text(9,16,'Identical for both',6.3,color=NEUTRAL)
 for j,n in enumerate([2,3,4,5]):p.text(9+31*(j+.3)/3.6,38,str(n),6,color=NEUTRAL,ha='center')
 p.text(9,42,'Perturbable blocks',6,color=NEUTRAL)
 a=p.ax(9,51,31,20)
 for sid,col in [('B4-007',BLOCK),('B4-029',NEUTRAL)]:
  d=C[C.id==sid].sort_values('q');a.plot(range(4),d['sum']/20,'-o',c=col,ms=3.5,mec='white',mew=.4,lw=1.2);yy=float(d['sum'].iloc[-1]/20);a.annotate(sid+'\n'+('20/20' if yy else '0/20'),(3,yy),xytext=(5,0),textcoords='offset points',ha='left',va='center',fontsize=6.2,color=col,weight='bold' if yy else 'normal',annotation_clip=False)
 a.set_xlim(-.3,3.3);a.set_ylim(-.08,1.1);a.set_yticks([0,.5,1]);a.set_xticks(range(4),['25','50','75','100']);a.set_xlabel('q (%)',labelpad=1);a.set_ylabel('Recovery',labelpad=1);a.spines['bottom'].set_color(HAIR)

def panel_E(p):
 p.title('E','Signature-specific inferential\nreversal fingerprint')
 p.text(5,11,'D blocked − D naive',6.3,color=NEUTRAL)
 d=read('E_fingerprint.tsv');div=LinearSegmentedColormap.from_list('contrast',[NAIVE,'#FAF9F6',BLOCK]);ids=[i for i in IDS if i!='FINAL-053'];x0=18;cw=8.1
 for j,col in enumerate(QC):p.text(x0+(j+.5)*cw,19,str(int(Q[j]*100)),6.1,color=col,ha='center',weight='bold')
 for i,sid in enumerate(ids):
  yy=25+i*6.6;p.text(16.5,yy+1.4,sid,6.1,ha='right')
  for j,q in enumerate(Q):
   v=float(d[(d.signature_id=='PF4C-'+sid)&(d.q==q)].contrast.iloc[0]);p.rect(x0+j*cw,yy,cw-.35,6.1,div((v+1)/2));p.text(x0+(j+.5)*cw,yy+1.4,f'{v:+.2f}' if v else '0.00',6.1,color='white' if abs(v)>.55 else INK,ha='center')
 a=p.ax(18,75,32,2);a.axis('off');a.set_xlim(-1,1);a.set_ylim(0,1)
 for j in range(100):a.add_patch(Rectangle((-1+j/50,0),.0202,1,fc=div(j/99),ec='none',antialiased=False))
 p.text(18,79,'−1',6,ha='center');p.text(34,79,'0',6,ha='center');p.text(50,79,'+1',6,ha='center')
 p.text(0,85,'Naive reveals more',6,color=NAIVE);p.text(0,90,'Blocked reveals more',6,color=BLOCK)


def panel_F(p):
 p.title('F','Inferential representation changes\nvisibility in both directions')
 comp=read('F_composition.tsv');a=p.ax(10,15,66,17)
 for j,row in enumerate(comp.itertuples(index=False,name=None)):
  q,*counts=row;left=0
  for n,col in zip(counts,[NEUTRAL,NAIVE,BLOCK,BOTH]):
   a.barh(j,n,left=left,height=.68,color=col,edgecolor='white',lw=.35)
   if n>=9:a.text(left+n/2,j,str(int(n)),ha='center',va='center',fontsize=6,color='white' if col!=NEUTRAL else INK)
   left+=n
 a.set_xlim(0,160);a.set_ylim(3.6,-.6);a.set_yticks(range(4),['q25','q50','q75','q100']);a.set_xticks([]);a.tick_params(axis='y',length=0,pad=3);a.spines[['left','bottom']].set_visible(False)
 for xx,txt,col in [(10,'Neither',NEUTRAL),(27,'Naive only',NAIVE),(46,'Blocked only',BLOCK),(69,'Both',BOTH)]:p.text(xx,12,txt,6,color=col)
 ax=p.ax(10,43,61,48);x=-np.log10(F.naive_p);y=-np.log10(F.blocked_p);cols={0:NEUTRAL,1:NAIVE,10:BLOCK,11:BOTH};collections=[]
 bounds={0:(0,0,TH,TH),1:(TH,0,3.2-TH,TH),10:(0,TH,TH,3.2-TH),11:(TH,TH,3.2-TH,3.2-TH)}
 for k,col in cols.items():
  m=F.REVERSAL_CELL==k;cmap=LinearSegmentedColormap.from_list('count_'+str(k),[tuple(.72*np.array(to_rgb('white'))+.28*np.array(to_rgb(col))),col]);hb=ax.hexbin(x[m],y[m],gridsize=(28,9),extent=(0,3.15,0,3.15),mincnt=1,cmap=cmap,linewidths=.25,edgecolors='white',zorder=2);collections.append(hb)
  clip=Rectangle(bounds[k][:2],*bounds[k][2:],transform=ax.transData);hb.set_clip_path(clip)
  for (xx,yy),n in zip(hb.get_offsets(),hb.get_array()):record('F','hexbin',reversal_cell=k,x=xx,y=yy,count=int(n))
  assert int(hb.get_array().sum())==int(m.sum())
 vmax=max(float(h.get_array().max()) for h in collections)
 for h in collections:h.set_norm(LogNorm(vmin=1,vmax=vmax))
 ax.axvline(TH,c='#7F7D7A',ls=(0,(3,2)),lw=.75,zorder=4);ax.axhline(TH,c='#7F7D7A',ls=(0,(3,2)),lw=.75,zorder=4);ax.plot([0,3.15],[0,3.15],c=HAIR,lw=.5,zorder=0)
 ax.set_xlim(-.015,3.15);ax.set_ylim(-.015,3.15);ax.set_xticks([0,1,2,3]);ax.set_yticks([0,1,2,3]);ax.set_xlabel('−log10(p naive)',labelpad=1);ax.set_ylabel('−log10(p blocked)',labelpad=1)
 for xx,yy,k,align,label in [(.025,.92,10,'left','blocked-only'),(.98,.92,11,'right','both'),(.395,.39,0,'right','neither'),(.98,.31,1,'right','naive-only')]:
  n=int((F.REVERSAL_CELL==k).sum());ax.text(xx,yy,str(n),transform=ax.transAxes,ha=align,va='top',fontsize=10.5,weight='bold',color='#65625E' if k==0 else cols[k]);ax.text(xx,yy-.145,f'{n/640:.1%}\n{label}',transform=ax.transAxes,ha=align,va='top',fontsize=6.1,color='#65625E' if k==0 else cols[k],linespacing=1.08)
 top=p.ax(10,37,61,5);right=p.ax(72,43,5,48);grid=np.linspace(0,3.15,220)
 for aa,vals,col,vert in [(top,x,NAIVE,False),(right,y,BLOCK,True)]:
  den=density(vals,grid)
  if vert:aa.fill_betweenx(grid,0,den,color=col,alpha=.12,lw=0);aa.plot(den,grid,c=col,lw=.7);aa.set_ylim(-.015,3.15)
  else:aa.fill_between(grid,den,color=col,alpha=.12,lw=0);aa.plot(grid,den,c=col,lw=.7);aa.set_xlim(-.015,3.15)
  aa.axis('off')
  for xx,yy in zip(grid,den):record('F','marginal',arm='blocked' if vert else 'naive',x=xx,y=yy)
 p.text(10,103,'640 worlds · bin count (log)\n80 α-unreachable worlds retained in neither',6.1,color=NEUTRAL,linespacing=1.1)

def panel_G(p):
 p.title('G','Same observed statistic, different admissible reference')
 p.text(105,0,'Dashed: null q95',6,color=NEUTRAL);p.text(136,0,'— Blocked',6.4,color=BLOCK);p.text(163,0,'— Naive',6.4,color=NAIVE)
 for j,(sid,title) in enumerate([('B4-007','Evidence restored by legal blocking'),('FINAL-079','Apparent evidence revoked by legal blocking')]):
  x0=j*98;p.text(x0+5,7,title,7.2,weight='bold');p.text(x0+5,12,f'{sid} · q100 · world 1',6.2,color=NEUTRAL)
  ax=p.ax(x0+10,19,81,31);d=G[G.id==sid];r=EX[EX.id==sid].iloc[0];lo=d['T'].min();hi=max(d['T'].max(),r.T_ALT);pad=.035*(hi-lo);grid=np.linspace(lo-pad,hi+pad,700)
  peak=0
  for arm,col in [('blocked',BLOCK),('naive',NAIVE)]:
   vals=d[d.arm==arm]['T'];den=density(vals,grid);peak=max(peak,den.max());ax.fill_between(grid,den,color=col,alpha=.27,lw=0);ax.plot(grid,den,c=col,lw=1.05);ax.axvline(vals.quantile(.95),c=col,ls=(0,(3,2)),lw=.75)
   for xx,yy in zip(grid,den):record('G','density',signature_id=sid,arm=arm,x=xx,y=yy)
  ax.axvline(r.T_ALT,c=INK,lw=1.3);ax.set_ylim(0,peak*1.12);ax.set_xlim(lo-pad,hi+pad);ax.set_yticks([]);ax.spines['left'].set_visible(False);ax.set_xlabel('Test statistic T',labelpad=1.5);ax.set_ylabel('Density',labelpad=2);ax.spines['bottom'].set_color(HAIR)
  ax.text(r.T_ALT,peak*1.13,'T ALT',fontsize=6,ha='center',va='bottom');ax.text(.98,.94,f'p blocked = {r.blocked_p:.3f}',transform=ax.transAxes,ha='right',va='top',fontsize=6.5,color=BLOCK);ax.text(.98,.81,f'p naive = {r.naive_p:.3f}',transform=ax.transAxes,ha='right',va='top',fontsize=6.5,color=NAIVE)


# Scale vertical mark spacing per panel, preserving all physical font sizes.
class ScaledPanel(Panel):
 def __init__(self,*args,sy=1,**kwargs):super().__init__(*args,**kwargs);self.sy=sy
 def ax(self,x,y,w,h):return super().ax(x,y*self.sy,w,h*self.sy)
 def text(self,x,y,s,*args,**kwargs):return super().text(x,y*self.sy,s,*args,**kwargs)
 def line(self,x1,y1,x2,y2,*args,**kwargs):return super().line(x1,y1*self.sy,x2,y2*self.sy,*args,**kwargs)
 # rect calls self.ax so y/h scaling already applies; rounded millimetres local.
REG={'A':(10,12,87,61,.86),'B':(104,12,96,61,.86),'C':(10,80,190,81,.75),'D':(10,168,50,73,.82),'E':(65,168,52,73,.78),'F':(122,168,78,73,.68),'G':(10,245,190,44,.69)}
FUN={l:globals()['panel_'+l] for l in REG}
def main():
 parser=argparse.ArgumentParser();parser.add_argument('--pass-name',default='final');args=parser.parse_args()
 assert (O/'Figure4_ACT4_updated_validation.csv').is_file(),'Run validation before rendering.'
 records.clear()
 for l,(x,y,w,h,sy) in REG.items():
  fig=plt.figure(figsize=((w+4)/25.4,(h+7)/25.4));FUN[l](ScaledPanel(fig,2,2,w,h,w+4,h+7,sy=sy))
  for ext in ['pdf','svg','png']:fig.savefig(O/'panels'/f'Figure4_panel_{l}.{ext}',dpi=180)
  plt.close(fig)
 records.clear();fig=plt.figure(figsize=(210/25.4,297/25.4))
 for l,(x,y,w,h,sy) in REG.items():FUN[l](ScaledPanel(fig,x,y,w,h,sy=sy))
 stem=O/'Figure4_ACT4_updated_A4';fig.savefig(str(stem)+'.pdf');fig.savefig(str(stem)+'.svg');fig.savefig(str(stem)+'_600dpi.png',dpi=600);fig.savefig(O/'QA'/f'{args.pass_name}_150dpi.png',dpi=150)
 plt.close(fig)
 # Full raw records + actual visual summaries; unused historical fields omitted.
 raw=[]
 for label,d,cols in [('B',A,list(A.columns)),('B',B,['signature_id','q','sum','size','fraction']),('C_D',C,['signature_id','q','sum','size','K_target','E_s_n','P_s_q_n','minimum_attainable_exact_p','reachable','recovery']),('E',E,['signature_id','median_ALT_T','median_blocked_null_T','median_blocked_null_q95','minimum_attainable_exact_p']),('F',F,['signature_id','q','replicate_id','T_ALT','blocked_p','naive_p','REVERSAL_CELL','minimum_attainable_exact_p']),('G',G,['signature_id','replicate_id','arm','draw','T','T_ALT','p'])]:
  t=d[cols].copy();t.insert(0,'panel',label);t.insert(1,'kind','raw');raw.append(t)
 raw=[t for t in raw if t['panel'].iloc[0] != 'E' and not (t['panel'].iloc[0]=='B' and 'fraction' in t.columns)]
 for label,name in [('A','A_landscape.tsv'),('B','B_envelope.tsv'),('E','E_fingerprint.tsv'),('F','F_composition.tsv')]:
  t=read(name);t.insert(0,'panel',label);t.insert(1,'kind','updated_raw');raw.append(t)
 raw.append(pd.DataFrame(records));pd.concat(raw,ignore_index=True).to_csv(O/'Figure4_ACT4_updated_data_used.csv',index=False)
 print('Rendered',args.pass_name,'; density/bin/summary records',len(records))
if __name__=='__main__':main()
