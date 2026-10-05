import cv2, numpy as np
K3=np.ones((3,3),np.uint8)
def color_mask(raw):
    mx=raw.max(2).astype(int); mn=raw.min(2).astype(int)
    return (((mx-mn)>=80)&(mx>=180)).astype(np.uint8)
def product_mask(g):
    b=cv2.GaussianBlur(g,(7,7),0)
    _,t=cv2.threshold(b,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
    n,lab,st,_=cv2.connectedComponentsWithStats(t)
    if n<2: return np.zeros_like(g,bool)
    k=1+np.argmax(st[1:,cv2.CC_STAT_AREA]); m=(lab==k).astype(np.uint8)
    return cv2.erode(m,np.ones((9,9),np.uint8))>0
def hp(g):
    g=g.astype(np.float32); return g-cv2.GaussianBlur(g,(5,5),1.2)
def process(raw, rng, sigma_scale=1.0, n_decoy=None):
    """raw BGR -> gray image with color overlay inpainted, same-shape decoys inpainted elsewhere, noise reinjected.
    returns img, boxes(xywh px of colour components), repair_mask, decoy boxes"""
    g=cv2.cvtColor(raw,cv2.COLOR_BGR2GRAY)
    cm=color_mask(raw); rep=cv2.dilate(cm,K3,iterations=1)
    n,lab,st,_=cv2.connectedComponentsWithStats(cm)
    allc=[(st[i,0],st[i,1],st[i,2],st[i,3]) for i in range(1,n) if st[i,cv2.CC_STAT_AREA]>=8]
    comps=[c for c in allc if 8<=c[2]<=60 and 8<=c[3]<=60]  # box-shaped overlay only; other colour marks are inpainted but not labelled
    prod=product_mask(g)
    H,W=g.shape; occupied=np.zeros_like(cm)
    for x,y,w,h in comps: occupied[max(0,y-6):y+h+6,max(0,x-6):x+w+6]=1
    decoys=[]; full=rep.copy()
    for i,(x,y,w,h) in enumerate(comps if n_decoy is None else comps[:n_decoy]):
        if W-w-3<=0 or H-h-3<=0: continue
        patch=rep[max(0,y-1):y+h+1,max(0,x-1):x+w+1]; ph,pw=patch.shape
        for _ in range(200):
            nx=int(rng.integers(0,W-pw)); ny=int(rng.integers(0,H-ph))
            if prod[ny:ny+ph,nx:nx+pw].all() and not occupied[max(0,ny-4):ny+ph+4,max(0,nx-4):nx+pw+4].any():
                full[ny:ny+ph,nx:nx+pw]|=patch; occupied[ny:ny+ph,nx:nx+pw]=1; decoys.append((nx+1,ny+1,w,h)); break
    # inpaint: real overlay (so the colour is removed) and decoy outline (trace at non-defect spot)
    img=cv2.inpaint(g,full,3,cv2.INPAINT_TELEA)
    # noise reinjection inside repaired pixels, matched to product noise level
    ring=(cv2.dilate(full,np.ones((7,7),np.uint8))>0)&~(full>0)&prod
    s=hp(g)[ring].std() if ring.sum()>50 else hp(g)[prod].std()
    f=img.astype(np.float32); m=full>0
    f[m]+=rng.normal(0,s*sigma_scale,m.sum())
    img=np.clip(np.round(f),0,255).astype(np.uint8)
    return img, comps, full, decoys
