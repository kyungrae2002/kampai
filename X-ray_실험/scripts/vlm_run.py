"""Qwen2.5-VL-3B-Instruct LoRA 파인튜닝(MPS) → 박스 JSON 출력 → 공통 예측 CSV.
사용: python vlm_run.py train   |  python vlm_run.py zeroshot  |  python vlm_run.py smoke"""
import sys, os, json, re, time, random, glob
import torch
from PIL import Image
from common import *
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
from peft import LoraConfig, get_peft_model, PeftModel
MODE=sys.argv[1]; MID='Qwen/Qwen2.5-VL-3B-Instruct'; SCALE=1.5
DEV=DEVICE if DEVICE!='mps' or torch.backends.mps.is_available() else 'cpu'; DT=torch.bfloat16
OUT=f'{RUNS}/qwen25vl3b_lora'; os.makedirs(OUT,exist_ok=True)
PROMPT=('This is a grayscale X-ray image of a product. Find every foreign object (small dark spot) inside the product. '
        'Return a JSON list like [{"bbox_2d":[x1,y1,x2,y2],"label":"defect"}] in pixel coordinates of this image. Return [] if there is none.')
def prep(p):
    im=Image.open(p).convert('RGB'); W,H=im.size
    w2=max(28,round(W*SCALE/28)*28); h2=max(28,round(H*SCALE/28)*28)
    return im.resize((w2,h2),Image.BICUBIC),(w2/W,h2/H)
def answer(p,d,sx,sy):
    W,H=Image.open(p).size; out=[]
    for l in open(f'{d}/labels/{stem(p)}.txt'):
        if l.strip():
            c,x,y,w,h=map(float,l.split())
            out.append({'bbox_2d':[round((x-w/2)*W*sx),round((y-h/2)*H*sy),round((x+w/2)*W*sx),round((y+h/2)*H*sy)],'label':'defect'})
    return json.dumps(out)
proc=AutoProcessor.from_pretrained(MID)  # 기본 max_pixels≈1M: 1.5배 확대 영상(최대 868x672)도 축소 없이 들어감
def msgs(ans=None):
    m=[{'role':'user','content':[{'type':'image'},{'type':'text','text':PROMPT}]}]
    if ans is not None: m.append({'role':'assistant','content':[{'type':'text','text':ans}]})
    return m
def encode(im,ans):
    full=proc.apply_chat_template(msgs(ans),tokenize=False)
    pr=proc.apply_chat_template(msgs(),tokenize=False,add_generation_prompt=True)
    x=proc(text=[full],images=[im],return_tensors='pt'); n=proc(text=[pr],images=[im],return_tensors='pt')['input_ids'].shape[1]
    lab=x['input_ids'].clone(); lab[:,:n]=-100; x['labels']=lab; return x
def sample_train(n_each):
    random.seed(0); S=[]
    for d,k in zip(TRAIN_DIRS,n_each):
        ims=images(d); random.shuffle(ims); S+= [(p,d) for p in ims[:k]]
    random.shuffle(S); return S
def load_base():
    return Qwen2_5_VLForConditionalGeneration.from_pretrained(MID, dtype=DT).to(DEV)
def parse(txt):
    try:
        j=json.loads(re.search(r'\[.*\]',txt,re.S).group(0)); return [b['bbox_2d'] for b in j if isinstance(b,dict) and len(b.get('bbox_2d',[]))==4]
    except Exception:
        return [list(map(float,m)) for m in re.findall(r'\[\s*(\d+\.?\d*)\s*,\s*(\d+\.?\d*)\s*,\s*(\d+\.?\d*)\s*,\s*(\d+\.?\d*)\s*\]',txt)]
@torch.no_grad()
def predict(model,name,limit=None):
    model.eval(); model.config.use_cache=True; rows=[]; raw=[]; lat=[]
    for sp,d in EVAL_DIRS.items():
        for p in (images(d)[:limit] if limit else images(d)):
            im,(sx,sy)=prep(p); pr=proc.apply_chat_template(msgs(),tokenize=False,add_generation_prompt=True)
            x=proc(text=[pr],images=[im],return_tensors='pt').to(DEV); t=time.time()
            o=model.generate(**x,max_new_tokens=160,do_sample=False); lat.append(time.time()-t)
            txt=proc.batch_decode(o[:,x['input_ids'].shape[1]:],skip_special_tokens=True)[0]; raw.append({'image':stem(p),'split':sp,'text':txt})
            for b in parse(txt):
                x1,y1,x2,y2=b; rows.append([stem(p),sp,f'{x1/sx:.2f}',f'{y1/sy:.2f}',f'{x2/sx:.2f}',f'{y2/sy:.2f}','1.0'])
    write_preds(f'{PRED}/{name}.csv',rows); json.dump(raw,open(f'{PRED}/{name}_raw.json','w'),ensure_ascii=False)
    return 1000*sum(lat)/max(1,len(lat))
if MODE=='zeroshot':
    ms=predict(load_base(),'qwen25vl3b_zeroshot'); json.dump({'model':'Qwen2.5-VL-3B zero-shot','ms_per_image_mps':ms},open(f'{PRED}/qwen25vl3b_zeroshot_meta.json','w')); print('DONE zeroshot'); sys.exit()
smoke=MODE=='smoke'
model=load_base(); model.config.use_cache=False; model.gradient_checkpointing_enable(); model.enable_input_require_grads()
cfg=LoraConfig(r=16,lora_alpha=32,lora_dropout=0.05,target_modules=r'^(?!.*visual).*\.(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)$',task_type='CAUSAL_LM')
model=get_peft_model(model,cfg); model.print_trainable_parameters()
# 원본 340 / 임의위치 합성 150 / 경계합성 150 / 전부제거(정상) 250 / 일부제거 150 / 약라벨 260  = 1,300
S=sample_train([4,2,2,2,2,2] if smoke else [340,150,150,250,150,260]); EPOCHS=1 if smoke else 2; ACC=8
opt=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=1e-4,weight_decay=0.0)
total=EPOCHS*len(S)//ACC; sched=torch.optim.lr_scheduler.LambdaLR(opt,lambda s: min(1,(s+1)/20)*max(0.05,1-s/max(1,total)))
model.train(); t0=time.time(); step=0; log=open(f'{LOGS}/vlm_train_loss.csv','a')
for ep in range(EPOCHS):
    random.shuffle(S)
    for i,(p,d) in enumerate(S):
        im,(sx,sy)=prep(p); x=encode(im,answer(p,d,sx,sy)).to(DEV)
        loss=model(**x).loss/ACC; loss.backward()
        if (i+1)%ACC==0:
            torch.nn.utils.clip_grad_norm_(model.parameters(),1.0); opt.step(); sched.step(); opt.zero_grad(); step+=1
            log.write(f'{ep},{step},{loss.item()*ACC:.4f},{time.time()-t0:.0f}\n'); log.flush()
            if step%10==0: print(f'ep{ep} step{step}/{total} loss {loss.item()*ACC:.4f} {time.time()-t0:.0f}s',flush=True)
train_sec=time.time()-t0
if not smoke: model.save_pretrained(OUT)
name='qwen25vl3b_lora'+('_smoke' if smoke else '')
ms=predict(model,name,limit=4 if smoke else None)
json.dump({'model':'Qwen2.5-VL-3B LoRA','train_samples':len(S),'epochs':EPOCHS,'train_seconds':train_sec,'ms_per_image_mps':ms},open(f'{PRED}/{name}_meta.json','w'),indent=1)
print('DONE',name)
