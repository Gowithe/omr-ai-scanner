from flask import Flask, render_template_string, request, redirect
import cv2
import json
import numpy as np
import base64
import os
from ultralytics import YOLO

# =====================================
# CONFIG
# =====================================

# โมเดล YOLO (1 class = X)
MODEL_PATH = r"runs/detect/train_AE52/weights/bestX.pt"

# ไฟล์ template (ต้องเตรียมไว้แล้ว)
TEMPLATE_FILES = {
    60: "questions_60.json",
    80: "questions_80.json",
    100: "questions_100.json",
}

OPTIONS = ["A", "B", "C", "D", "E"]

TARGET_WIDTH = 1600
TARGET_HEIGHT = 2300

CONF_THRES = 0.10
MAX_SLOT_DIST = 70.0

UPLOAD_IMAGE_PATH = "uploaded_sheet.jpg"

# ถ้าไม่กรอกเฉลย จะใช้ตัวนี้แทน (ปล่อยว่างไว้ก็ได้)
ANSWER_KEY_DEFAULT = {
    # 1: "A",
}

# =====================================
# LOAD YOLO MODEL
# =====================================

print(f"โหลดโมเดลจาก: {MODEL_PATH}")
model = YOLO(MODEL_PATH)

# =====================================
# TEMPLATE & MAPPING
# =====================================

def load_template(num_questions: int):
    """โหลด template json -> dict {slot_index(int): (x,y)}"""
    if num_questions not in TEMPLATE_FILES:
        raise ValueError(f"ยังไม่ได้ตั้งค่า template สำหรับ {num_questions} ข้อ")

    json_path = TEMPLATE_FILES[num_questions]
    if not os.path.exists(json_path):
        raise FileNotFoundError(f"ไม่พบไฟล์ template: {json_path}")

    with open(json_path, "r", encoding="utf-8") as f:
        raw = json.load(f)
        all_slots = {int(k): (v["x"], v["y"]) for k, v in raw.items()}

    print(
        f"โหลด template สำหรับ {num_questions} ข้อ "
        f"จาก {json_path} | จำนวนจุดทั้งหมด: {len(all_slots)}"
    )
    return all_slots


def build_slot_mapping(all_slots: dict, num_questions: int):
    """
    แปลงจาก slot_index (key ใน template) → (ข้อ, ตัวเลือก A–E)

    - กรณี 60 ข้อ: ใช้ลำดับ index ตรง ๆ แบบเดิม
      index ตามลำดับ 1..5 = ข้อ 1 (A–E)
      6..10 = ข้อ 2
      ...
    - กรณี 80 / 100 ข้อ: ใช้การจัดกลุ่มตามแถว y แล้วเรียง x ซ้าย→ขวา
    """

    # ---------- 60 ข้อ: ใช้วิธี index-based ----------
    if num_questions == 60:
        mapping = {}
        sorted_indices = sorted(all_slots.keys())
        for pos, idx in enumerate(sorted_indices):
            qnum = pos // 5 + 1
            opt_i = pos % 5
            if 1 <= qnum <= num_questions:
                mapping[idx] = (qnum, OPTIONS[opt_i])
        print(f"[60Q] สร้าง slot_mapping แบบ index แล้ว {len(mapping)} ช่อง")
        return mapping

    # ---------- 80 / 100 ข้อ: ใช้วิธีจัดแถวตาม y ----------
    slots = [(idx, xy[0], xy[1]) for idx, xy in all_slots.items()]
    slots.sort(key=lambda t: t[2])  # sort ตาม y

    if len(slots) == 0:
        raise ValueError("template ไม่มีจุดเลย")

    y_vals = [s[2] for s in slots]
    dy = [y_vals[i + 1] - y_vals[i] for i in range(len(y_vals) - 1)]
    median_dy = np.median(dy) if dy else 1.0
    row_tol = max(5.0, float(median_dy) / 2.0)

    rows = []
    current_row = [slots[0]]
    current_y = slots[0][2]

    for idx, x, y in slots[1:]:
        if abs(y - current_y) <= row_tol:
            current_row.append((idx, x, y))
            current_y = (current_y * (len(current_row) - 1) + y) / len(current_row)
        else:
            rows.append(current_row)
            current_row = [(idx, x, y)]
            current_y = y
    rows.append(current_row)

    print(f"[{num_questions}Q] ตรวจพบแถวทั้งหมด {len(rows)} แถว")

    mapping = {}
    qnum = 1

    for row_i, row in enumerate(rows, start=1):
        row_sorted = sorted(row, key=lambda t: t[1])  # sort ตาม x

        if len(row_sorted) != 5:
            print(f"⚠ แถวที่ {row_i} มี {len(row_sorted)} ช่อง (คาด 5)")

        for opt_i, opt in enumerate(OPTIONS):
            if opt_i >= len(row_sorted):
                break
            idx, x, y = row_sorted[opt_i]
            if qnum <= num_questions:
                mapping[idx] = (qnum, opt)

        qnum += 1
        if qnum > num_questions:
            break

    print(f"[{num_questions}Q] สร้าง slot_mapping แล้ว {len(mapping)} ช่อง (คาด {num_questions*5})")
    return mapping

# =====================================
# WARP & GEOMETRY
# =====================================

def order_points(pts):
    pts = np.array(pts, dtype="float32")
    s = pts.sum(axis=1)
    tl = pts[np.argmin(s)]
    br = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    tr = pts[np.argmin(diff)]
    bl = pts[np.argmax(diff)]
    return np.array([tl, tr, br, bl], dtype="float32")


def warp_from_four_points(image, pts):
    """warp จาก 4 มุม → ภาพขนาด TARGET_WIDTH / TARGET_HEIGHT"""
    rect = order_points(pts)
    dst = np.array(
        [
            [0, 0],
            [TARGET_WIDTH - 1, 0],
            [TARGET_WIDTH - 1, TARGET_HEIGHT - 1],
            [0, TARGET_HEIGHT - 1],
        ],
        dtype="float32",
    )
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (TARGET_WIDTH, TARGET_HEIGHT))
    cv2.imwrite("debug_warp_from_click.jpg", warped)
    return warped


def find_nearest_slot(xc, yc, all_slots, max_dist: float = MAX_SLOT_DIST):
    """หา slot index ที่ใกล้ (xc,yc) ที่สุดใน template"""
    best_idx = None
    best_d2 = max_dist * max_dist

    for idx, (sx, sy) in all_slots.items():
        dx = xc - sx
        dy = yc - sy
        d2 = dx * dx + dy * dy
        if d2 < best_d2:
            best_d2 = d2
            best_idx = idx

    return best_idx

# =====================================
# YOLO → ANSWERS (เตรียมข้อมูลแถวสำหรับ 80/100)
# =====================================

def prepare_row_info(all_slots, slot_mapping):
    """
    สร้างข้อมูลสำหรับแบบ 80/100 ข้อ:
      - row_center_y[q]  = ค่า y เฉลี่ยของแถวนั้น (ใช้เลือกข้อจาก y)
      - row_midpoints[q] = จุดกึ่งกลางระหว่าง x ของ A–B, B–C, C–D, D–E
      - row_tol          = tolerance สำหรับเลือกแถวจาก y
    """
    row_y_vals = {}
    row_x_vals = {}

    for slot_idx, (q, opt) in slot_mapping.items():
        x, y = all_slots[slot_idx]
        row_y_vals.setdefault(q, []).append(y)
        row_x_vals.setdefault(q, []).append(x)

    row_center_y = {}
    row_midpoints = {}

    for q, ys in row_y_vals.items():
        row_center_y[q] = float(np.mean(ys))

        xs = sorted(row_x_vals[q])
        if len(xs) < 2:
            continue

        mids = []
        for i in range(len(xs) - 1):
            mids.append((xs[i] + xs[i + 1]) / 2.0)
        row_midpoints[q] = mids

    y_list = sorted(row_center_y.values())
    if len(y_list) >= 2:
        dy = [y_list[i + 1] - y_list[i] for i in range(len(y_list) - 1)]
        median_dy = np.median(dy)
    else:
        median_dy = 40.0

    row_tol = max(5.0, float(median_dy) / 2.0)

    print(f"เตรียม row_info: {len(row_center_y)} แถว, row_tol={row_tol:.2f}")
    return row_center_y, row_midpoints, row_tol

# =====================================
# YOLO → ANSWERS (หลัก)
# =====================================

def read_answers_from_image_bgr(img_bgr, num_questions: int,
                                all_slots, slot_mapping,
                                conf_thres: float = CONF_THRES):
    """
    YOLO (1 class X) → จุด X
    ใช้ template + slot_mapping แปลงเป็น (ข้อ, ตัวเลือก)
    รวมผลต่อข้อ -> answers[q] = 'A'/'B'/.../'MULTI'/None

    - 60 ข้อ: ใช้ nearest-slot แบบเดิม
    - 80/100 ข้อ: ใช้แถวจาก y + midpoints x และ bias C→D ถ้ากากบาทล้ำช่อง D
    """

    results = model.predict(source=img_bgr, conf=conf_thres, verbose=False)
    det = results[0]

    answers = {q: None for q in range(1, num_questions + 1)}

    if det.boxes is None or len(det.boxes) == 0:
        print("❗ YOLO ไม่เจอกากบาทเลย")
        return answers, img_bgr

    boxes = det.boxes.xyxy.cpu().numpy()
    confs = det.boxes.conf.cpu().numpy()
    print(f"YOLO เจอกากบาททั้งหมด {len(boxes)} จุด")

    debug_img = img_bgr.copy()

    marks_by_q = {q: {} for q in range(1, num_questions + 1)}

    if num_questions > 60:
        row_center_y, row_midpoints, row_tol = prepare_row_info(all_slots, slot_mapping)
    else:
        row_center_y, row_midpoints, row_tol = None, None, None

    for (x1, y1, x2, y2), conf in zip(boxes, confs):
        conf = float(conf)
        xc = (x1 + x2) / 2.0
        yc = (y1 + y2) / 2.0

        if num_questions == 60:
            slot_idx = find_nearest_slot(xc, yc, all_slots, max_dist=MAX_SLOT_DIST)
            if slot_idx is None:
                continue
            qopt = slot_mapping.get(slot_idx)
            if qopt is None:
                continue
            qnum, opt = qopt
        else:
            best_q = None
            best_dy = row_tol
            for q, cy in row_center_y.items():
                dyq = abs(yc - cy)
                if dyq <= best_dy:
                    best_dy = dyq
                    best_q = q

            if best_q is None:
                continue

            mids = row_midpoints.get(best_q)
            if not mids or len(mids) < 4:
                slot_idx = find_nearest_slot(xc, yc, all_slots, max_dist=MAX_SLOT_DIST)
                if slot_idx is None:
                    continue
                qopt = slot_mapping.get(slot_idx)
                if qopt is None:
                    continue
                qnum, opt = qopt
            else:
                if xc < mids[0]:
                    opt_idx = 0
                elif xc < mids[1]:
                    opt_idx = 1
                elif xc < mids[2]:
                    opt_idx = 2
                elif xc < mids[3]:
                    opt_idx = 3
                else:
                    opt_idx = 4

                mid_cd = mids[2]
                if opt_idx == 2 and x2 > mid_cd:
                    opt_idx = 3

                qnum = best_q
                opt = OPTIONS[opt_idx]

        prev_conf = marks_by_q[qnum].get(opt, 0.0)
        if conf > prev_conf:
            marks_by_q[qnum][opt] = conf

        cv2.putText(
            debug_img,
            f"{qnum}{opt} {conf:.2f}",
            (int(xc), int(yc)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 255),
            2,
        )

    for q in range(1, num_questions + 1):
        opts_conf = marks_by_q[q]
        if not opts_conf:
            continue
        if len(opts_conf) == 1:
            answers[q] = max(opts_conf.items(), key=lambda x: x[1])[0]
        else:
            answers[q] = "MULTI"

    return answers, debug_img

# =====================================
# GRADING
# =====================================

def grade_answers(answers: dict, answer_key: dict, num_questions: int):
    correct = 0
    total = 0
    detail = {}
    blank = 0
    multi = 0
    wrong = 0

    for q in range(1, num_questions + 1):
        correct_opt = answer_key.get(q)
        if correct_opt is None:
            continue

        total += 1
        stu_ans = answers.get(q)

        if stu_ans is None:
            detail[q] = ("-", None, correct_opt)
            blank += 1
        elif stu_ans == "MULTI":
            detail[q] = ("M", stu_ans, correct_opt)
            multi += 1
        elif stu_ans == correct_opt:
            detail[q] = ("✔", stu_ans, correct_opt)
            correct += 1
        else:
            detail[q] = ("✘", stu_ans, correct_opt)
            wrong += 1

    stats = {
        "correct": correct,
        "wrong": wrong,
        "blank": blank,
        "multi": multi,
        "total": total,
    }
    return correct, total, detail, stats


def parse_answer_key_string(s: str, num_questions: int):
    """"AAAABBBCC..." -> dict {1:'A',2:'A',...}"""
    s_clean = "".join(ch.upper() for ch in s if ch.upper() in OPTIONS)
    key = {}
    for i, ch in enumerate(s_clean, start=1):
        if i > num_questions:
            break
        key[i] = ch
    return key

# =====================================
# FLASK + HTML (UI สวยขึ้น)
# =====================================

app = Flask(__name__)

TEMPLATE_UPLOAD = """
<!doctype html>
<html lang="th">
<head>
  <meta charset="utf-8">
  <title>OMR AI Scanner</title>
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <style>
    :root {
      --bg: #020617;
      --card: #020617;
      --card2:#020617;
      --border: #1f2937;
      --accent: #22c55e;
      --accent2:#38bdf8;
      --text: #e5e7eb;
      --muted: #9ca3af;
      --danger:#ef4444;
    }
    * { box-sizing:border-box; }
    body {
      margin:0;
      font-family: system-ui, -apple-system, "Segoe UI", sans-serif;
      background:
        radial-gradient(circle at 0% 0%, rgba(56,189,248,0.12), transparent 55%),
        radial-gradient(circle at 100% 100%, rgba(34,197,94,0.15), transparent 55%),
        #020617;
      color:var(--text);
      min-height:100vh;
      display:flex;
      align-items:center;
      justify-content:center;
      padding:16px;
    }
    .shell {
      width:100%;
      max-width:960px;
      display:grid;
      grid-template-columns: minmax(0,1.1fr) minmax(0,0.9fr);
      gap:24px;
      align-items:stretch;
    }
    @media (max-width: 860px) {
      .shell { grid-template-columns: minmax(0,1fr); }
    }
    .hero {
      padding:24px 20px;
      border-radius:24px;
      background: radial-gradient(circle at 15% 0%, rgba(56,189,248,0.28), transparent 55%), #020617;
      border:1px solid rgba(148,163,184,0.35);
      position:relative;
      overflow:hidden;
    }
    .hero h1 {
      font-size:2.1rem;
      margin:0 0 6px;
      letter-spacing:0.06em;
      text-transform:uppercase;
      color:var(--accent2);
    }
    .hero h2 {
      margin:0;
      font-size:1.1rem;
      color:var(--muted);
      font-weight:400;
    }
    .chip-row { margin-top:14px; display:flex; gap:8px; flex-wrap:wrap; }
    .chip {
      padding:4px 10px;
      border-radius:999px;
      border:1px solid rgba(148,163,184,0.4);
      font-size:0.75rem;
      color:var(--muted);
      backdrop-filter: blur(8px);
    }
    .stats {
      margin-top:18px;
      display:flex;
      gap:16px;
      flex-wrap:wrap;
    }
    .stat-box {
      flex:1 1 120px;
      border-radius:18px;
      padding:10px 12px;
      background:linear-gradient(135deg,rgba(15,23,42,0.9),rgba(15,23,42,0.4));
      border:1px solid rgba(15,23,42,0.9);
    }
    .stat-label {
      font-size:0.75rem;
      color:var(--muted);
    }
    .stat-value {
      margin-top:4px;
      font-size:1.1rem;
      font-weight:600;
    }
    .step-list { margin-top:18px; font-size:0.9rem; }
    .step {
      display:flex;
      gap:10px;
      align-items:flex-start;
      margin-bottom:6px;
      color:var(--muted);
    }
    .step .circle {
      width:18px;
      height:18px;
      border-radius:999px;
      border:1px solid rgba(148,163,184,0.7);
      display:flex;
      align-items:center;
      justify-content:center;
      font-size:0.7rem;
      color:var(--accent2);
    }

    .card {
      background:var(--card2);
      border-radius:24px;
      border:1px solid var(--border);
      padding:20px 18px 18px;
      box-shadow:0 24px 60px rgba(15,23,42,0.9);
    }
    .card h3 {
      margin:0 0 4px;
      font-size:1.1rem;
      display:flex;
      align-items:center;
      gap:6px;
    }
    .card h3 span.dot {
      width:10px; height:10px; border-radius:999px;
      background:var(--accent);
      box-shadow:0 0 0 4px rgba(34,197,94,0.4);
    }
    .card p {
      margin:0 0 12px;
      font-size:0.85rem;
      color:var(--muted);
    }
    label {
      display:block;
      margin-bottom:6px;
      font-size:0.85rem;
    }
    .field {
      margin-bottom:12px;
    }
    input[type=file],
    input[type=text],
    select {
      width:100%;
      padding:9px 10px;
      border-radius:10px;
      border:1px solid var(--border);
      background:#020617;
      color:var(--text);
      font-size:0.9rem;
    }
    input[type=text]::placeholder {
      color:#6b7280;
    }
    button {
      width:100%;
      margin-top:6px;
      padding:11px 12px;
      border-radius:999px;
      border:none;
      background:linear-gradient(135deg,#22c55e,#16a34a);
      color:white;
      font-size:0.95rem;
      font-weight:600;
      cursor:pointer;
      box-shadow:0 16px 35px rgba(22,163,74,0.5);
    }
    button:hover {
      filter:brightness(1.05);
    }
    .hint {
      margin-top:8px;
      font-size:0.78rem;
      color:var(--muted);
    }
    .badge-variant {
      display:inline-flex;
      align-items:center;
      gap:5px;
      font-size:0.75rem;
      background:rgba(15,23,42,0.95);
      padding:4px 9px;
      border-radius:999px;
      border:1px solid rgba(148,163,184,0.5);
    }
    .badge-dot {
      width:6px;height:6px;border-radius:999px;background:var(--accent2);
    }
  </style>
</head>
<body>
  <div class="shell">
    <section class="hero">
      <h1>OMR AI Scanner</h1>
      <h2>ตรวจข้อสอบเลือกตอบจากกระดาษคำตอบ 60 / 80 / 100 ข้อ</h2>

      <div class="chip-row">
        <span class="chip">⚡ YOLO-based detection</span>
        <span class="chip">🎯 จุดกาเพี้ยนเล็กน้อยก็อ่านได้</span>
        <span class="chip">📝 มีภาพ debug ให้ตรวจสอบ</span>
      </div>

      <div class="stats">
        <div class="stat-box">
          <div class="stat-label">รองรับจำนวนข้อ</div>
          <div class="stat-value">60 · 80 · 100</div>
        </div>
        <div class="stat-box">
          <div class="stat-label">รูปแบบคำตอบ</div>
          <div class="stat-value">A – E</div>
        </div>
        <div class="stat-box">
          <div class="stat-label">เหมาะสำหรับ</div>
          <div class="stat-value">สอบกลางภาค / ปลายภาค</div>
        </div>
      </div>

      <div class="step-list">
        <div class="step">
          <div class="circle">1</div>
          <div>อัปโหลดรูปกระดาษคำตอบทั้งแผ่น</div>
        </div>
        <div class="step">
          <div class="circle">2</div>
          <div>เลือกจำนวนข้อ + ใส่เฉลยถ้ามี</div>
        </div>
        <div class="step">
          <div class="circle">3</div>
          <div>คลิก 4 มุมกระดาษ → ระบบจะทำการ warp และตรวจให้อัตโนมัติ</div>
        </div>
      </div>
    </section>

    <section class="card">
      <h3><span class="dot"></span> เริ่มต้นตรวจข้อสอบ</h3>
      <p>เลือกไฟล์รูปถ่ายกระดาษคำตอบ แล้วตั้งค่าด้านล่าง จากนั้นกดปุ่มเพื่อไปขั้นตอนเลือกมุมกระดาษ</p>

      <form method="post" action="/select" enctype="multipart/form-data">
        <div class="field">
          <label>📷 รูปกระดาษคำตอบ</label>
          <input type="file" name="sheet" accept="image/*" capture="environment" required>
        </div>

        <div class="field">
          <label>📄 จำนวนข้อในชุดนี้</label>
          <select name="num_questions">
            <option value="60">60 ข้อ</option>
            <option value="80">80 ข้อ</option>
            <option value="100">100 ข้อ</option>
          </select>
        </div>

        <div class="field">
          <label>
            ✏️ เฉลย (ถ้ามี)
            <span class="badge-variant" style="margin-left:4px;">
              <span class="badge-dot"></span> ตัวอย่าง: AAAABBBBCCCC...
            </span>
          </label>
          <input type="text" name="answer_key" placeholder="ใส่เฉลยเรียงตามข้อ 1–N หรือเว้นว่างถ้าจะดูเฉพาะคำตอบนักเรียน">
        </div>

        <button type="submit">ไปขั้นตอนที่ 2: เลือกมุมกระดาษ</button>

        <div class="hint">
          แนะนำให้ถ่ายให้เห็นกระดาษทั้งแผ่น ชัด ๆ ไม่มีเงามืด และไม่เอียงมากเกินไป
        </div>
      </form>
    </section>
  </div>
</body>
</html>
"""

TEMPLATE_SELECT = """
<!doctype html>
<html lang="th">
<head>
  <meta charset="utf-8">
  <title>เลือก 4 มุมกระดาษคำตอบ</title>
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <style>
    :root {
      --bg:#020617;
      --card:#020617;
      --border:#1f2937;
      --accent:#22c55e;
      --accent2:#38bdf8;
      --text:#e5e7eb;
      --muted:#9ca3af;
    }
    * { box-sizing:border-box; }
    body {
      margin:0;
      font-family: system-ui, -apple-system, "Segoe UI", sans-serif;
      background:
        radial-gradient(circle at 0% 0%, rgba(56,189,248,0.15), transparent 55%),
        radial-gradient(circle at 100% 100%, rgba(34,197,94,0.18), transparent 55%),
        #020617;
      color:var(--text);
      min-height:100vh;
      display:flex;
      align-items:center;
      justify-content:center;
      padding:16px;
    }
    .wrap {
      width:100%;
      max-width:980px;
    }
    .card {
      background:var(--card);
      border-radius:24px;
      border:1px solid var(--border);
      padding:18px 18px 20px;
      box-shadow:0 26px 70px rgba(15,23,42,0.95);
    }
    h1 {
      margin:0 0 6px;
      font-size:1.6rem;
      text-align:left;
      color:var(--accent2);
    }
    .subtitle {
      margin:0 0 12px;
      font-size:0.9rem;
      color:var(--muted);
    }
    .grid {
      display:grid;
      grid-template-columns:minmax(0,1.2fr) minmax(0,0.9fr);
      gap:18px;
      align-items:flex-start;
    }
    @media (max-width:880px){
      .grid { grid-template-columns:minmax(0,1fr); }
    }
    img {
      max-width:100%;
      height:auto;
      display:block;
      border-radius:18px;
      border:1px solid #111827;
      background:#020617;
    }
    .side {
      font-size:0.86rem;
    }
    .pill {
      display:inline-flex;
      align-items:center;
      gap:6px;
      padding:4px 10px;
      border-radius:999px;
      border:1px solid rgba(148,163,184,0.55);
      font-size:0.78rem;
      color:var(--muted);
      margin-bottom:8px;
    }
    .dot {
      width:8px;height:8px;border-radius:999px;background:var(--accent);
    }
    .steps { margin-top:6px; }
    .step {
      display:flex;
      gap:8px;
      align-items:flex-start;
      margin-bottom:4px;
      color:var(--muted);
    }
    .step-num {
      width:18px;height:18px;border-radius:999px;
      border:1px solid rgba(148,163,184,0.8);
      display:flex;align-items:center;justify-content:center;
      font-size:0.75rem;color:var(--accent2);
    }
    .points-box {
      margin-top:10px;
      padding:8px 10px;
      border-radius:12px;
      background:#020617;
      border:1px dashed rgba(148,163,184,0.7);
      font-size:0.8rem;
      color:var(--muted);
      min-height:40px;
    }
    button {
      width:100%;
      margin-top:10px;
      padding:11px 12px;
      border-radius:999px;
      border:none;
      background:linear-gradient(135deg,#22c55e,#16a34a);
      color:white;
      font-size:0.95rem;
      font-weight:600;
      cursor:pointer;
      box-shadow:0 18px 45px rgba(22,163,74,0.55);
    }
    button:disabled {
      opacity:0.45;
      box-shadow:none;
      cursor:not-allowed;
    }
    .meta {
      margin-top:6px;
      font-size:0.8rem;
      color:var(--muted);
    }
  </style>
</head>
<body>
  <div class="wrap">
    <div class="card">
      <h1>ขั้นตอนที่ 2 · เลือก 4 มุมกระดาษ</h1>
      <p class="subtitle">
        คลิกที่รูปกระดาษคำตอบตามลำดับ
        <b>ซ้ายบน → ขวาบน → ขวาล่าง → ซ้ายล่าง</b>
        เพื่อให้ระบบจัดมุมและปรับภาพให้ตรง
      </p>

      <div class="grid">
        <div>
          <img id="sheetImage" src="data:image/jpeg;base64,{{ img_data }}" alt="sheet">
        </div>

        <div class="side">
          <div class="pill"><span class="dot"></span> จำนวนข้อในชุดนี้: {{ num_questions }} ข้อ</div>

          <div class="steps">
            <div class="step">
              <div class="step-num">1</div>
              <div>ซูมดูให้แน่ใจว่ากระดาษเห็นครบทั้ง 4 มุม</div>
            </div>
            <div class="step">
              <div class="step-num">2</div>
              <div>คลิกทีละจุด: <b>ซ้ายบน → ขวาบน → ขวาล่าง → ซ้ายล่าง</b></div>
            </div>
            <div class="step">
              <div class="step-num">3</div>
              <div>เมื่อครบ 4 จุด ปุ่ม “ตรวจข้อสอบตอนนี้” จะกดได้</div>
            </div>
          </div>

          <div class="points-box">
            <b>จุดที่เลือก:</b>
            <span id="pointsDisplay">ยังไม่ได้เลือก</span>
          </div>

          <form method="post" action="/grade">
            <input type="hidden" id="pointsInput" name="points">
            <input type="hidden" name="answer_key" value="{{ answer_key_str }}">
            <input type="hidden" name="num_questions" value="{{ num_questions }}">
            <button type="submit" id="submitBtn" disabled>ตรวจข้อสอบตอนนี้</button>
          </form>

          <div class="meta">
            ถ้าเลือกผิด สามารถรีเฟรชหน้า (F5) เพื่อเริ่มคลิกใหม่ได้
          </div>
        </div>
      </div>
    </div>
  </div>

  <script>
    const img = document.getElementById('sheetImage');
    const pointsDisplay = document.getElementById('pointsDisplay');
    const pointsInput = document.getElementById('pointsInput');
    const submitBtn = document.getElementById('submitBtn');
    let clicks = [];

    img.addEventListener('click', function(e) {
      const rect = img.getBoundingClientRect();
      const x = e.clientX - rect.left;
      const y = e.clientY - rect.top;

      const natW = img.naturalWidth;
      const natH = img.naturalHeight;
      const dispW = img.clientWidth;
      const dispH = img.clientHeight;

      const scaleX = natW / dispW;
      const scaleY = natH / dispH;

      const origX = Math.round(x * scaleX);
      const origY = Math.round(y * scaleY);

      if (clicks.length >= 4) return;
      clicks.push([origX, origY]);

      const names = ["ซ้ายบน", "ขวาบน", "ขวาล่าง", "ซ้ายล่าง"];
      pointsDisplay.textContent = clicks.map((p, i) =>
        names[i] + ` (${p[0]}, ${p[1]})`
      ).join("  ·  ");

      if (clicks.length === 4) {
        pointsInput.value = clicks.map(p => p[0] + "," + p[1]).join(";");
        submitBtn.disabled = false;
      }
    });
  </script>
</body>
</html>
"""

TEMPLATE_RESULT = """
<!doctype html>
<html lang="th">
<head>
  <meta charset="utf-8">
  <title>ผลการตรวจข้อสอบ</title>
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <style>
    :root {
      --bg:#020617;
      --card:#020617;
      --border:#1f2937;
      --accent:#22c55e;
      --accent2:#38bdf8;
      --danger:#f97373;
      --text:#e5e7eb;
      --muted:#9ca3af;
    }
    * { box-sizing:border-box; }
    body {
      margin:0;
      font-family: system-ui, -apple-system, "Segoe UI", sans-serif;
      background:
        radial-gradient(circle at 0% 0%, rgba(56,189,248,0.16), transparent 55%),
        radial-gradient(circle at 100% 100%, rgba(34,197,94,0.16), transparent 55%),
        #020617;
      color:var(--text);
      min-height:100vh;
      display:flex;
      align-items:center;
      justify-content:center;
      padding:16px;
    }
    .wrap { width:100%; max-width:1040px; }
    .card {
      background:var(--card);
      border-radius:24px;
      border:1px solid var(--border);
      padding:20px 18px 20px;
      box-shadow:0 28px 75px rgba(15,23,42,0.95);
    }
    h1 {
      margin:0 0 6px;
      font-size:1.6rem;
      color:var(--accent2);
    }
    .subtitle {
      margin:0 0 12px;
      font-size:0.9rem;
      color:var(--muted);
    }
    .layout {
      display:grid;
      grid-template-columns:minmax(0,1.05fr) minmax(0,1fr);
      gap:18px;
      align-items:flex-start;
    }
    @media (max-width:880px){
      .layout { grid-template-columns:minmax(0,1fr); }
    }
    .score-card {
      border-radius:18px;
      padding:12px 12px 10px;
      background:radial-gradient(circle at 0 0, rgba(34,197,94,0.4), transparent 55%), #020617;
      border:1px solid rgba(34,197,94,0.5);
      margin-bottom:10px;
    }
    .score-main {
      font-size:2.2rem;
      font-weight:700;
    }
    .score-main span {
      font-size:1.2rem;
      font-weight:500;
      color:var(--muted);
    }
    .score-row {
      margin-top:6px;
      display:flex;
      gap:10px;
      flex-wrap:wrap;
      font-size:0.82rem;
    }
    .pill {
      padding:3px 9px;
      border-radius:999px;
      border:1px solid rgba(148,163,184,0.6);
      color:var(--muted);
      display:inline-flex;
      align-items:center;
      gap:6px;
    }
    .pill-ok { border-color:rgba(34,197,94,0.8); color:#bbf7d0; }
    .pill-bad{ border-color:rgba(239,68,68,0.8); color:#fecaca; }
    .pill-dot {
      width:7px;height:7px;border-radius:999px;background:var(--accent);
    }
    table {
      width:100%;
      border-collapse:collapse;
      font-size:0.8rem;
      margin-top:8px;
      background:#020617;
      border-radius:14px;
      overflow:hidden;
    }
    th, td {
      border-bottom:1px solid #111827;
      padding:4px 6px;
      text-align:center;
    }
    th {
      background:#020617;
      color:var(--muted);
      font-weight:500;
    }
    tr:nth-child(even) td {
      background:#020617;
    }
    tr:last-child td {
      border-bottom:none;
    }
    .status-ok { color:#4ade80; }
    .status-wrong { color:#f97373; }
    .status-multi { color:#fde047; }
    .status-blank { color:#9ca3af; }
    img {
      max-width:100%;
      border-radius:16px;
      border:1px solid #111827;
      margin-top:8px;
      display:block;
    }
    .section-title {
      margin-top:10px;
      margin-bottom:4px;
      font-size:0.95rem;
    }
    .btn-row {
      margin-top:10px;
      display:flex;
      flex-wrap:wrap;
      gap:8px;
    }
    a.button {
      display:inline-flex;
      align-items:center;
      justify-content:center;
      padding:9px 16px;
      border-radius:999px;
      background:#1d4ed8;
      color:white;
      text-decoration:none;
      font-size:0.85rem;
      font-weight:500;
    }
    a.button.outline {
      background:transparent;
      border:1px solid #374151;
      color:var(--muted);
    }
    .small-note {
      margin-top:4px;
      font-size:0.78rem;
      color:var(--muted);
    }
  </style>
</head>
<body>
  <div class="wrap">
    <div class="card">
      <h1>ผลการตรวจข้อสอบ ({{ num_questions }} ข้อ)</h1>
      <p class="subtitle">
        ระบบได้ตรวจคำตอบจากกระดาษคำตอบที่อัปโหลด พร้อมแสดงรายละเอียดและภาพตำแหน่งที่ AI ตรวจพบ
      </p>

      <div class="layout">
        <div>
          <div class="score-card">
            <div class="score-main">
              {{ stats.correct }} / {{ stats.total }}
              <span>ข้อที่มีเฉลย</span>
            </div>
            <div class="score-row">
              <span class="pill pill-ok"><span class="pill-dot"></span> ถูก {{ stats.correct }} ข้อ</span>
              <span class="pill pill-bad">ผิด {{ stats.wrong }} ข้อ</span>
              <span class="pill">เว้นว่าง {{ stats.blank }} ข้อ</span>
              <span class="pill">ตอบหลายตัว {{ stats.multi }} ข้อ</span>
            </div>
          </div>

          <h3 class="section-title">รายละเอียดรายข้อ</h3>
          <table>
            <tr>
              <th>ข้อ</th>
              <th>คำตอบนักเรียน</th>
              <th>เฉลย</th>
              <th>สถานะ</th>
            </tr>
            {% for q in range(1, num_questions+1) %}
            <tr>
              <td>{{ q }}</td>
              <td>{{ answers[q] if answers[q] else "-" }}</td>
              <td>{{ answer_key.get(q, "-") }}</td>
              <td>
                {% if q in detail %}
                  {% set mark = detail[q][0] %}
                  {% if mark == "✔" %}
                    <span class="status-ok">✔ ถูก</span>
                  {% elif mark == "✘" %}
                    <span class="status-wrong">✘ ผิด</span>
                  {% elif mark == "M" %}
                    <span class="status-multi">M หลายตัว</span>
                  {% else %}
                    <span class="status-blank">- เว้นว่าง</span>
                  {% endif %}
                {% else %}
                  <span class="status-blank">-</span>
                {% endif %}
              </td>
            </tr>
            {% endfor %}
          </table>

          <div class="btn-row">
            <a href="/" class="button">ตรวจชุดใหม่</a>
          </div>
          <div class="small-note">
            สามารถเซฟหน้านี้เป็น PDF หรือแคปหน้าจอเก็บไว้เป็นหลักฐานการตรวจได้
          </div>
        </div>

        <div>
          {% if debug_image %}
          <h3 class="section-title">ภาพ Debug · จุดที่ AI ตรวจพบ</h3>
          <img src="data:image/jpeg;base64,{{ debug_image }}" alt="debug image">
          <div class="small-note">
            ตัวหนังสือเหลืองบนกระดาษคือผลตรวจจากโมเดล (หมายเลขข้อ + ตัวเลือก + ค่าเชื่อมั่น)
          </div>
          {% endif %}
        </div>
      </div>
    </div>
  </div>
</body>
</html>
"""

# =====================================
# ROUTES
# =====================================

@app.route("/", methods=["GET"])
def upload_page():
    return render_template_string(TEMPLATE_UPLOAD)


@app.route("/select", methods=["POST"])
def select_corners():
    file = request.files.get("sheet")
    if not file:
        return "ไม่พบไฟล์รูป", 400

    num_questions = int(request.form.get("num_questions", "60"))
    answer_key_str = request.form.get("answer_key", "").strip()

    file_bytes = np.frombuffer(file.read(), np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    if img is None:
        return "อ่านรูปไม่ได้", 400

    cv2.imwrite(UPLOAD_IMAGE_PATH, img)

    _, buf = cv2.imencode(".jpg", img)
    img_b64 = base64.b64encode(buf).decode("utf-8")

    return render_template_string(
        TEMPLATE_SELECT,
        img_data=img_b64,
        answer_key_str=answer_key_str,
        num_questions=num_questions,
    )


@app.route("/grade", methods=["POST"])
def grade():
    if not os.path.exists(UPLOAD_IMAGE_PATH):
        return redirect("/")

    points_str = request.form.get("points", "").strip()
    if not points_str:
        return "ยังไม่ได้เลือก 4 มุม", 400

    num_questions = int(request.form.get("num_questions", "60"))

    try:
        all_slots = load_template(num_questions)
        slot_mapping = build_slot_mapping(all_slots, num_questions)
    except Exception as e:
        return f"ปัญหาในการโหลด template: {e}", 500

    pts = []
    for p in points_str.split(";"):
        x_str, y_str = p.split(",")
        pts.append([float(x_str), float(y_str)])
    pts = np.array(pts, dtype="float32")
    if pts.shape != (4, 2):
        return "รูปแบบจุดไม่ถูกต้อง", 400

    img = cv2.imread(UPLOAD_IMAGE_PATH)
    if img is None:
        return "โหลดรูปไม่สำเร็จ", 500

    warped = warp_from_four_points(img, pts)

    answers, debug_img = read_answers_from_image_bgr(
        warped, num_questions, all_slots, slot_mapping
    )

    raw_key = request.form.get("answer_key", "").strip()
    if raw_key:
        effective_key = parse_answer_key_string(raw_key, num_questions)
    else:
        effective_key = ANSWER_KEY_DEFAULT

    if effective_key:
        _, _, detail, stats = grade_answers(answers, effective_key, num_questions)
    else:
        blank = sum(1 for v in answers.values() if v is None)
        multi = sum(1 for v in answers.values() if v == "MULTI")
        answered = num_questions - blank
        stats = {
            "correct": 0,
            "wrong": 0,
            "blank": blank,
            "multi": multi,
            "total": answered,
        }
        detail = {}

    _, buf = cv2.imencode(".jpg", debug_img)
    debug_b64 = base64.b64encode(buf).decode("utf-8")

    return render_template_string(
        TEMPLATE_RESULT,
        answers=answers,
        stats=stats,
        detail=detail,
        debug_image=debug_b64,
        num_questions=num_questions,
        answer_key=effective_key,
    )


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
