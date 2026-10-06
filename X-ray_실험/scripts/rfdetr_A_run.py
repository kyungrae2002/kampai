"""RF-DETR-S를 A 구성(실제 영상만, ablation/A_실제만.txt 2,220장)으로 재학습 → 검증·테스트·정상합성·스트레스셋 예측.
학습 조건은 기존 rfdetr_s(D 구성)와 동일: 15ep, batch 4 × 누적 4, lr 1e-4.
출력: runs/rfdetr_s_A, preds/rfdetr_s_A.csv, preds/stress_rfdetr_s_A.csv, preds/rfdetr_s_A_meta.json"""
import os, glob, json, time
from common import *
from PIL import Image
def main():
    CO = f'{ROOT}/rfdetr_data_A'
    def build(split, paths):
        o = f'{CO}/{split}'; os.makedirs(o, exist_ok=True)
        js = {'images': [], 'annotations': [], 'categories': [{'id': 0, 'name': 'defects', 'supercategory': 'none'}, {'id': 1, 'name': 'defect', 'supercategory': 'defects'}]}
        aid = 0
        for p in paths:
            s = stem(p); fn = s + '.png'; dst = f'{o}/{fn}'
            if not os.path.lexists(dst): os.symlink(os.path.relpath(p, o), dst)
            W, H = Image.open(p).size; iid = len(js['images']); js['images'].append({'id': iid, 'file_name': fn, 'width': W, 'height': H})
            lab = os.path.join(os.path.dirname(os.path.dirname(p)), 'labels', s + '.txt')
            for l in open(lab):
                if not l.strip(): continue
                c, x, y, w, h = map(float, l.split()); bw, bh = w*W, h*H
                js['annotations'].append({'id': aid, 'image_id': iid, 'category_id': 1, 'bbox': [x*W-bw/2, y*H-bh/2, bw, bh], 'area': bw*bh, 'iscrowd': 0}); aid += 1
        json.dump(js, open(f'{o}/_annotations.coco.json', 'w')); print(split, len(js['images']), 'images', aid, 'boxes')
    train = [l.strip() for l in open(f'{ROOT}/ablation/A_실제만.txt') if l.strip()]
    train = [p for p in train if stem(p) not in EXCLUDE]
    build('train', train); build('valid', images(EVAL_DIRS['val'])); build('test', images(EVAL_DIRS['test']))
    from rfdetr import RFDETRSmall
    out = f'{RUNS}/rfdetr_s_A'; os.makedirs(out, exist_ok=True)
    kw = dict(num_workers=2, dataset_dir=CO, epochs=15, batch_size=4, grad_accum_steps=4, lr=1e-4, output_dir=out)
    m = RFDETRSmall(); t0 = time.time()
    try: m.train(device=DEVICE, **kw)
    except TypeError as e:
        print('device kwarg not accepted, retrying without it:', e); m.train(**kw)
    train_sec = time.time() - t0
    ck = [f for f in ['checkpoint_best_total.pth', 'checkpoint_best_ema.pth', 'checkpoint_best_regular.pth', 'checkpoint.pth'] if os.path.exists(f'{out}/{f}')]
    print('checkpoint used:', ck[0])
    best = RFDETRSmall(pretrain_weights=f'{out}/{ck[0]}', num_classes=1)
    try: best.optimize_for_inference()
    except Exception as e: print('optimize skipped', e)
    def pred(dirs):
        rows = []; lat = []
        for sp, d in dirs.items():
            for p in images(d):
                im = Image.open(p).convert('RGB'); t = time.time(); det = best.predict(im, threshold=0.001); lat.append(time.time() - t)
                for (x1, y1, x2, y2), s in zip(det.xyxy, det.confidence):
                    rows.append([stem(p), sp, f'{x1:.2f}', f'{y1:.2f}', f'{x2:.2f}', f'{y2:.2f}', f'{s:.5f}'])
        return rows, lat
    rows, lat = pred(EVAL_DIRS); write_preds(f'{PRED}/rfdetr_s_A.csv', rows)
    srows, _ = pred({os.path.basename(d): d for d in sorted(glob.glob(f'{ROOT}/eval_sets/stress_*'))}); write_preds(f'{PRED}/stress_rfdetr_s_A.csv', srows)
    json.dump({'model': 'RF-DETR-S (A)', 'checkpoint': ck[0], 'train_images': len(train), 'train_seconds': train_sec, 'epochs_requested': 15,
               'ms_per_image_mps_batch1': 1000*sum(lat[5:])/max(1, len(lat)-5)}, open(f'{PRED}/rfdetr_s_A_meta.json', 'w'), ensure_ascii=False, indent=1)
    print('DONE rfdetr_s_A')
if __name__ == '__main__':  # macOS DataLoader spawn 대비
    main()
