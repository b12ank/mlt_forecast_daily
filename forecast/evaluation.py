"""Versioned evaluation of saved wide predictions; never trains a model."""
import base64
from html import escape
import io
import json

import numpy as np
import pandas as pd

HORIZONS = list(range(2, 8))
DATE_COLUMNS = ["predict_time", "D+2", "D+7"]
PRED_COLUMNS = DATE_COLUMNS + [f"pred_D+{h}" for h in HORIZONS]
TRUE_COLUMNS = DATE_COLUMNS + [f"true_D+{h}" for h in HORIZONS]
SMAPE_NOTE = (
    "SMAPE（%）= 100 × mean(2 × |预测值−真实值| / (|预测值|+|真实值|))；"
    "数值 20 表示 20%，范围 0～200%，越低越好。使用原始价格，不做小于 50 的截断；"
    "真实值和预测值同时为 0 时，该点误差记为 0。仅统计有效样本，无有效样本时为空。"
)


def smape_values(predicted, actual):
    """Pointwise SMAPE in percent for finite pairs; no floor or epsilon."""
    p, y = np.asarray(predicted, dtype=float), np.asarray(actual, dtype=float)
    denominator = np.abs(p) + np.abs(y)
    return 200 * np.divide(np.abs(p-y), denominator,
                           out=np.zeros_like(denominator), where=denominator != 0)


def validate_wide(frame, prefix):
    expected = PRED_COLUMNS if prefix == "pred" else TRUE_COLUMNS
    if list(frame) != expected:
        raise ValueError(f"结果表列必须为 {expected}")
    result = frame.copy()
    for col in DATE_COLUMNS:
        parsed = pd.to_datetime(result[col], format="%Y-%m-%d", errors="raise")
        if parsed.isna().any() or not parsed.dt.strftime("%Y-%m-%d").eq(result[col]).all():
            raise ValueError("日期必须为 YYYY-MM-DD")
    origins = pd.to_datetime(result["predict_time"])
    if origins.duplicated().any() or not origins.is_monotonic_increasing:
        raise ValueError("预测起点必须唯一且升序")
    for h in (2, 7):
        if not (origins + pd.Timedelta(days=h)).dt.strftime("%Y-%m-%d").eq(result[f"D+{h}"]).all():
            raise ValueError("目标日与预测起点错位")
    for col in expected[3:]:
        result[col] = pd.to_numeric(result[col], errors="raise")
        if np.isinf(result[col].to_numpy(dtype=float)).any():
            raise ValueError("预测或真实值包含无穷大")
    return result


def arrays(predictions, actuals):
    p, a = validate_wide(predictions, "pred"), validate_wide(actuals, "true")
    if not p[DATE_COLUMNS].reset_index(drop=True).equals(a[DATE_COLUMNS].reset_index(drop=True)):
        raise ValueError("预测与真实值的日期不一致")
    return p.iloc[:, 3:].to_numpy(dtype=float), a.iloc[:, 3:].to_numpy(dtype=float)


def signs(values, tolerance):
    return np.where(np.abs(values) <= tolerance, 0, np.sign(values)).astype(int)


def turns(values, tolerance):
    direction = signs(np.diff(values, axis=1), tolerance)
    left, right = direction[:, :-1], direction[:, 1:]
    return np.where((left > 0) & (right < 0), 1,
                    np.where((left < 0) & (right > 0), -1, 0))


def evaluate(predictions, actuals, tolerance=1e-8, common_mask=None):
    p, y = arrays(predictions, actuals)
    valid = np.isfinite(p) & np.isfinite(y)
    if common_mask is not None:
        if common_mask.shape != valid.shape:
            raise ValueError("共同样本掩码形状不一致")
        valid &= common_mask
    rows = []

    def add(scope, metric, value, count):
        rows.append({"scope": scope, "metric": metric, "value": value, "count": int(count)})

    for scope, indices in [("overall", list(range(6)))] + [(f"D+{h}", [h-2]) for h in HORIZONS]:
        v = valid[:, indices]
        errors = (p[:, indices] - y[:, indices])[v]
        add(scope, "MAE", float(np.abs(errors).mean()) if errors.size else None, errors.size)
        add(scope, "MSE", float(np.square(errors).mean()) if errors.size else None, errors.size)
        smape = smape_values(p[:, indices][v], y[:, indices][v])
        add(scope, "SMAPE", float(smape.mean()) if smape.size else None, smape.size)
    add("overall", "prediction_coverage", float(np.isfinite(p).mean()) if p.size else None, p.size)
    add("overall", "scored_coverage", float(valid.mean()) if valid.size else None, valid.size)
    complete = valid.all(axis=1)
    n = int(complete.sum())
    add("curve", "complete_coverage", float(complete.mean()) if len(complete) else None, len(complete))
    pc, yc = p[complete], y[complete]
    ordering = [signs(pc[:, j]-pc[:, i], tolerance) == signs(yc[:, j]-yc[:, i], tolerance)
                for i in range(6) for j in range(i+1, 6)]
    add("curve", "ordering_accuracy", float(np.mean(ordering)) if n else None, n*15)
    directions = signs(np.diff(pc, axis=1), tolerance) == signs(np.diff(yc, axis=1), tolerance)
    add("curve", "direction_accuracy", float(directions.mean()) if n else None, n*5)
    pt, yt = turns(pc, tolerance), turns(yc, tolerance)
    tp = int(((pt == yt) & (yt != 0)).sum())
    predicted, truth = int((pt != 0).sum()), int((yt != 0).sum())
    add("curve", "turn_precision", tp/predicted if predicted else None, predicted)
    add("curve", "turn_recall", tp/truth if truth else None, truth)
    add("curve", "turn_F1", 2*tp/(predicted+truth) if predicted+truth else None, predicted+truth)
    for metric, value in [("turn_true_positives", tp), ("turn_predicted", predicted), ("turn_actual", truth)]:
        add("curve", metric, value, n*4)
    return pd.DataFrame(rows)


def write_report(path, predictions, actuals, metrics, title, note=""):
    """Self-contained HTML with a selectable six-day SVG and an error histogram."""
    p, y = arrays(predictions, actuals)
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib import pyplot as plt
    fig, ax = plt.subplots(figsize=(7, 2.6))
    errors = (p-y)[np.isfinite(p) & np.isfinite(y)]
    if len(errors):
        ax.hist(errors, bins=40, color="#287b91")
    else:
        ax.text(0.5, 0.5, "No observed labels", ha="center", transform=ax.transAxes)
    ax.set(xlabel="Prediction - actual", ylabel="Count")
    fig.tight_layout()
    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", dpi=120)
    plt.close(fig)
    png = base64.b64encode(buffer.getvalue()).decode("ascii")
    data = []
    for i, origin in enumerate(predictions["predict_time"]):
        data.append({"origin": origin, "dates": [(pd.Timestamp(origin)+pd.Timedelta(days=h)).strftime("%Y-%m-%d") for h in HORIZONS],
                     "pred": [float(v) if np.isfinite(v) else None for v in p[i]],
                     "actual": [float(v) if np.isfinite(v) else None for v in y[i]]})
    payload = json.dumps(data, ensure_ascii=False, allow_nan=False).replace("<", "\\u003c")
    script = """
const data=PAYLOAD, select=document.getElementById('origin');
data.forEach((r,i)=>{const o=document.createElement('option');o.value=i;o.textContent=r.origin;select.appendChild(o);});
function draw(){
const r=data[Number(select.value)]; if(!r)return;
const vals=[...r.pred,...r.actual].filter(v=>v!==null); let lo=Math.min(...vals),hi=Math.max(...vals);
if(!vals.length){lo=0;hi=1;} if(hi===lo){lo-=1;hi+=1;}
const x=i=>70+i*132, y=v=>230-(v-lo)/(hi-lo)*180;
let svg='<rect width="800" height="290" fill="white"/>';
for(let i=0;i<5;i++){const v=lo+(hi-lo)*i/4,yy=y(v);svg+=`<line x1="65" x2="745" y1="${yy}" y2="${yy}" stroke="#ddd"/><text x="4" y="${yy+4}" font-size="12">${v.toFixed(2)}</text>`;}
for(const [key,color] of [['pred','#1479ab'],['actual','#d65c34']]){
for(let i=0;i<6;i++){const v=r[key][i];if(v===null)continue;
if(i>0&&r[key][i-1]!==null)svg+=`<line x1="${x(i-1)}" y1="${y(r[key][i-1])}" x2="${x(i)}" y2="${y(v)}" stroke="${color}" stroke-width="2"/>`;
svg+=`<circle cx="${x(i)}" cy="${y(v)}" r="4" fill="${color}"><title>${key}: ${v}</title></circle>`;
}}
r.dates.forEach((d,i)=>{svg+=`<text x="${x(i)}" y="265" text-anchor="middle" font-size="12">${d}</text>`;});
document.getElementById('curve').innerHTML=svg;
document.getElementById('values').textContent=r.dates.map((d,i)=>`${d}　预测 ${r.pred[i]??'缺失'}　真实 ${r.actual[i]??'未知'}`).join('\\n');
}
select.addEventListener('change',draw);draw();
""".replace("PAYLOAD", payload)
    page = f"""<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>{escape(title)}</title>
<style>body{{font:15px system-ui;max-width:1050px;margin:35px auto;padding:0 20px;color:#253341}}table{{border-collapse:collapse;width:100%}}td,th{{padding:7px;border-bottom:1px solid #ddd;text-align:left}}svg{{width:100%;height:auto}}pre{{white-space:pre-wrap}}.note{{background:#f0f5f7;padding:14px}}</style>
<h1>{escape(title)}</h1><p class="note">{escape(note)}</p>
<label>预测起点 <select id="origin"></select></label><p>蓝色：预测；橙色：真实</p>
<svg id="curve" viewBox="0 0 800 290"></svg><pre id="values"></pre>
<h2>误差分布</h2><img alt="误差分布" src="data:image/png;base64,{png}" style="max-width:100%">
<h2>评价指标</h2><p>{escape(SMAPE_NOTE)}</p>{metrics.to_html(index=False, na_rep='—', escape=True)}<script>{script}</script></html>"""
    path.write_text(page, encoding="utf-8")
