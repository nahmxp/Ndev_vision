"""From-scratch multi-scale template matching; no pretrained model."""
from dataclasses import dataclass
import numpy as np
from PIL import Image
@dataclass(frozen=True)
class MatchResult:
    score:float; error:float; x:int; y:int; width:int; height:int; scale:float; comparisons:int
def prepare_image(image,max_side=600):
    image=image.convert("L")
    if max(image.size)>max_side:
        r=max_side/max(image.size); image=image.resize((round(image.width*r),round(image.height*r)),Image.Resampling.LANCZOS)
    return image
def _features(image,mode):
    a=np.asarray(image,dtype=np.float32)/255
    if mode=="pixels": return a
    gy,gx=np.gradient(a); a=np.hypot(gx,gy)
    return a/a.max() if a.max() else a
def match_template(search_image,reference_image,*,min_scale=.5,max_scale=1.5,scale_steps=11,stride=4,mode="pixels",progress=None):
    if stride<1 or min_scale<=0 or max_scale<min_scale or scale_steps<1: raise ValueError("Invalid scale range or stride.")
    search,reference=prepare_image(search_image),prepare_image(reference_image); source=_features(search,mode); jobs=[]
    for scale in np.linspace(min_scale,max_scale,scale_steps):
        w,h=max(1,round(reference.width*scale)),max(1,round(reference.height*scale))
        if w<=search.width and h<=search.height:
            target=_features(reference.resize((w,h),Image.Resampling.LANCZOS),mode)
            xs=list(range(0,search.width-w+1,stride)); ys=list(range(0,search.height-h+1,stride))
            if xs[-1]!=search.width-w: xs.append(search.width-w)
            if ys[-1]!=search.height-h: ys.append(search.height-h)
            jobs.append((float(scale),w,h,target,xs,ys))
    if not jobs: raise ValueError("No scaled reference fits inside the search image.")
    total=sum(len(xs)*len(ys) for *_,xs,ys in jobs); done=0; best_error=float("inf"); best=None
    for scale,w,h,target,xs,ys in jobs:
        for y in ys:
            for x in xs:
                error=float(np.mean(np.abs(source[y:y+h,x:x+w]-target)))
                if error<best_error: best_error,best=error,(x,y,w,h,scale)
                done+=1
                if progress and (done==total or done%max(1,total//200)==0): progress(done,total,scale,1-best_error)
    x,y,w,h,scale=best
    return MatchResult(1-best_error,best_error,x,y,w,h,scale,done)
