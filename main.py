"""
HRMS Face ID v3 — YuNet (yuz topish) + SFace (neyron tarmoq bilan yuz tanish)
Aniqlik uchun yaxshilanishlar: sifat filtri, yorug'likni tekislash (CLAHE), ko'p kadrli o'rtacha,
ikkinchi eng yaqin xodim bilan farq (margin), xilma-xil namunalar, dublikat tekshiruvi.

pip install opencv-contrib-python pillow numpy
MUHIM: v2 da saqlangan xodimlarni qayta ro'yxatdan o'tkazing (embeddings/ papkasini tozalang),
chunki v3 da tasvir CLAHE bilan qayta ishlanadi va namunalar soni boshqacha.
"""
import csv
import json
import os
import time
import tkinter as tk
import urllib.request
from collections import deque
from datetime import datetime
from tkinter import messagebox, ttk

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageTk

# ----------------------------- SOZLAMALAR -----------------------------
BASE = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE, "models")
DATA_DIR = os.path.join(BASE, "embeddings")
EMP_FILE = os.path.join(BASE, "employees.json")
LOG_FILE = os.path.join(BASE, "attendance.csv")

CAM_INDEX = 0
VIDEO_W, VIDEO_H = 640, 480

# --- Ro'yxatdan o'tish ---
SAMPLES = 30              # nechta namuna (har biri uchun ko'zgu nusxasi ham saqlanadi)
DIVERSE_SIM_FRONT = 0.97  # to'g'ri qarashda: avvalgi namunaga shundan kam o'xshash bo'lsin
DIVERSE_SIM_TURN = 0.92   # burilganda: yanada xilma-xil bo'lsin
DUP_THRESHOLD = 0.60      # yangi yuz mavjud xodimga shundan ko'p o'xshasa, ogohlantiradi

# --- Tanish ---
MATCH_THRESHOLD = 0.45    # o'xshashlik chegarasi. Boshqa odam tanilsa oshiring (0.50), o'zingizni tanimasa tushiring (0.40)
HRMS_FACE_THRESHOLD = 0.70
MARGIN = 0.06             # eng yaqin va ikkinchi eng yaqin xodim orasidagi minimal farq
TOPK = 5                  # har xodim uchun eng yaxshi K namuna o'rtachasi
SMOOTH_FRAMES = 5         # oxirgi nechta kadr embeddingi o'rtacha olinadi
CONFIRM_FRAMES = 5        # ketma-ket nechta kadrda tanilsa, Check-in bo'ladi
COOLDOWN_SEC = 60
SHOW_SCORE = True

# --- Sifat filtri ---
MIN_FACE_PX = 90
MIN_DET_SCORE = 0.85      # YuNet ishonchi
MIN_SHARPNESS = 25.0      # xiralik chegarasi (Laplacian dispersiyasi)
MIN_BRIGHT, MAX_BRIGHT = 55, 215
MAX_YAW_RECOG = 0.45      # tanishda yon tomonga burilish chegarasi
MAX_YAW_FRONT = 0.20      # ro'yxatdan o'tishda to'g'ri qarash chegarasi
MAX_YAW_TURN = 0.70

SFACE_FILE = "face_recognition_sface_2021dec.onnx"
YUNET_FILES = ["face_detection_yunet_2023mar.onnx", "face_detection_yunet_2026may.onnx"]

BG = "#0f172a"
CARD = "#1e293b"
TEXT = "#e2e8f0"
MUTED = "#94a3b8"
ACCENT = "#3b82f6"
GREEN = (34, 197, 94)
RED = (239, 68, 68)
BLUE = (59, 130, 246)
AMBER = (245, 158, 11)

_CLAHE = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))


def hexc(rgb):
    return "#%02x%02x%02x" % rgb


def load_font(size):
    for name in ("segoeui.ttf", "arial.ttf", "Arial.ttf", "DejaVuSans.ttf",
                 "/System/Library/Fonts/Helvetica.ttc"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default()


def enhance(frame):
    """Yorug'lik farqlarini tekislaydi (CLAHE) — qorong'i/yorqin xonada ham barqaror."""
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    return cv2.cvtColor(cv2.merge((_CLAHE.apply(l), a, b)), cv2.COLOR_LAB2BGR)


def normalize(v):
    v = np.asarray(v, dtype=np.float32).reshape(-1)
    return v / (np.linalg.norm(v) + 1e-9)


# ----------------------------- MODELLAR -----------------------------
def model_urls(sub, fname):
    return [
        f"https://github.com/opencv/opencv_zoo/raw/main/models/{sub}/{fname}",
        f"https://huggingface.co/opencv/opencv_zoo/resolve/main/models/{sub}/{fname}",
    ]


def ensure_model(fname, sub):
    os.makedirs(MODEL_DIR, exist_ok=True)
    path = os.path.join(MODEL_DIR, fname)
    if os.path.exists(path) and os.path.getsize(path) > 20_000:
        return path
    for url in model_urls(sub, fname):
        try:
            print(f"Yuklanmoqda: {fname} ...")
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=120) as r, open(path, "wb") as f:
                f.write(r.read())
            if os.path.getsize(path) > 20_000:
                return path
            os.remove(path)
        except Exception as e:
            print("  xato:", e)
    return None


def get_haar():
    for path in (os.path.join(getattr(cv2, "data", None) and cv2.data.haarcascades or "",
                              "haarcascade_frontalface_default.xml"),
                 os.path.join(BASE, "haarcascade_frontalface_default.xml")):
        if os.path.exists(path):
            c = cv2.CascadeClassifier(path)
            if not c.empty():
                return c
    return None


# ----------------------------- MA'LUMOTLAR -----------------------------
def load_employees():
    data = {}
    if os.path.exists(EMP_FILE):
        with open(EMP_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
    return {u: n for u, n in data.items()
            if os.path.exists(os.path.join(DATA_DIR, f"{u}.npy"))}


def save_employees(data):
    with open(EMP_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def write_log(uid, name):
    new = not os.path.exists(LOG_FILE)
    now = datetime.now()
    with open(LOG_FILE, "a", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        if new:
            w.writerow(["sana", "vaqt", "id", "ism"])
        w.writerow([now.strftime("%Y-%m-%d"), now.strftime("%H:%M:%S"), uid, name])


def read_today_log():
    if not os.path.exists(LOG_FILE):
        return []
    today = datetime.now().strftime("%Y-%m-%d")
    with open(LOG_FILE, "r", newline="", encoding="utf-8-sig") as f:
        rows = [r for r in csv.reader(f)][1:]
    return [r for r in rows if len(r) >= 4 and r[0] == today][::-1]


# ----------------------------- ILOVA -----------------------------
class App:
    def __init__(self, root):
        self.root = root
        self.ok = False
        root.title("HRMS Face ID")
        root.configure(bg=BG)
        root.resizable(False, False)

        if not self.init_models():
            root.destroy()
            return

        os.makedirs(DATA_DIR, exist_ok=True)
        self.employees = load_employees()
        self.gallery = {}
        self.load_gallery()

        self.font_md = load_font(22)
        self.font_bn = load_font(26)

        self.mode = "recognize"
        self.candidate = None
        self.streak = 0
        self.recent = deque(maxlen=SMOOTH_FRAMES)
        self.last_logged = {}
        self.banner = None
        self.reg_name = ""
        self.reg_embs = []
        self.reg_flips = []
        self.reg_last = 0.0
        self.photo = None

        self.build_ui()
        self.refresh_lists()

        self.cap = cv2.VideoCapture(CAM_INDEX)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, VIDEO_W)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, VIDEO_H)
        if not self.cap.isOpened():
            messagebox.showerror("Xatolik", "Kamerani ochib bo'lmadi!")
            root.destroy()
            return

        root.protocol("WM_DELETE_WINDOW", self.on_close)
        self.ok = True
        self.loop()

    # ---------- Modellarni tayyorlash ----------
    def init_models(self):
        self.sface = None
        self.yunet = None
        self.haar = None

        sface_path = ensure_model(SFACE_FILE, "face_recognition_sface")
        if not sface_path:
            messagebox.showerror(
                "Model yuklanmadi",
                f"{SFACE_FILE} yuklab bo'lmadi.\n\nQo'lda yuklab, 'models' papkasiga tashlang:\n"
                + model_urls("face_recognition_sface", SFACE_FILE)[0])
            return False
        try:
            self.sface = cv2.FaceRecognizerSF.create(sface_path, "")
        except Exception as e:
            messagebox.showerror("Xatolik", f"SFace modelini ochib bo'lmadi:\n{e}")
            return False

        for fname in YUNET_FILES:
            path = ensure_model(fname, "face_detection_yunet")
            if not path:
                continue
            try:
                det = cv2.FaceDetectorYN.create(path, "", (VIDEO_W, VIDEO_H), 0.8, 0.3, 5000)
                det.setInputSize((VIDEO_W, VIDEO_H))
                det.detect(np.zeros((VIDEO_H, VIDEO_W, 3), np.uint8))
                self.yunet = det
                print("Detektor: YuNet", fname)
                break
            except Exception as e:
                print("YuNet ishlamadi:", fname, e)

        if self.yunet is None:
            self.haar = get_haar()
            if self.haar is None:
                messagebox.showerror(
                    "Xatolik", "Yuz topuvchi model yuklanmadi.\nInternetni tekshirib qayta urining.")
                return False
            print("Detektor: Haar (zaxira)")
        return True

    def load_gallery(self):
        self.gallery = {}
        for uid in self.employees:
            self.gallery[uid] = np.load(os.path.join(DATA_DIR, f"{uid}.npy"))

    # ---------- UI ----------
    def build_ui(self):
        style = ttk.Style()
        style.theme_use("clam")
        style.configure("Accent.TButton", background=ACCENT, foreground="white",
                        font=("Segoe UI", 10, "bold"), padding=8, borderwidth=0)
        style.map("Accent.TButton", background=[("active", "#2563eb")])
        style.configure("Danger.TButton", background="#475569", foreground="white",
                        font=("Segoe UI", 9), padding=6, borderwidth=0)
        style.map("Danger.TButton", background=[("active", "#dc2626")])
        style.configure("Treeview", background=CARD, fieldbackground=CARD,
                        foreground=TEXT, rowheight=26, borderwidth=0,
                        font=("Segoe UI", 10))
        style.configure("Treeview.Heading", background="#334155", foreground=TEXT,
                        font=("Segoe UI", 9, "bold"), borderwidth=0)
        style.map("Treeview", background=[("selected", ACCENT)])
        style.configure("Horizontal.TProgressbar", troughcolor="#334155",
                        background=hexc(GREEN), borderwidth=0)

        left = tk.Frame(self.root, bg=BG)
        left.grid(row=0, column=0, padx=(20, 10), pady=20, sticky="n")
        right = tk.Frame(self.root, bg=BG, width=380)
        right.grid(row=0, column=1, padx=(10, 20), pady=20, sticky="n")

        head = tk.Frame(left, bg=BG)
        head.pack(fill="x", pady=(0, 10))
        tk.Label(head, text="HRMS Face ID", bg=BG, fg=TEXT,
                 font=("Segoe UI", 20, "bold")).pack(side="left")
        self.clock_var = tk.StringVar()
        tk.Label(head, textvariable=self.clock_var, bg=BG, fg=MUTED,
                 font=("Segoe UI", 11)).pack(side="right")

        self.video = tk.Label(left, bg="black", bd=0,
                              highlightthickness=2, highlightbackground="#334155")
        self.video.pack()

        self.status_var = tk.StringVar(value="Kameraga qarang")
        self.status_lbl = tk.Label(left, textvariable=self.status_var, bg=CARD, fg=MUTED,
                                   font=("Segoe UI", 13, "bold"), pady=12)
        self.status_lbl.pack(fill="x", pady=(10, 0))

        reg = self.card(right, "Yangi xodim qo'shish")
        self.name_var = tk.StringVar()
        self.entry = tk.Entry(reg, textvariable=self.name_var, font=("Segoe UI", 12),
                              bg="#0f172a", fg=TEXT, insertbackground=TEXT,
                              relief="flat", highlightthickness=1,
                              highlightbackground="#475569", highlightcolor=ACCENT)
        self.entry.pack(fill="x", ipady=7, pady=(0, 8))
        self.entry.bind("<Return>", lambda e: self.start_register())
        self.reg_btn = ttk.Button(reg, text="Yuzni skaner qilish", style="Accent.TButton",
                                  command=self.start_register)
        self.reg_btn.pack(fill="x")
        self.progress = ttk.Progressbar(reg, maximum=SAMPLES)
        self.cancel_btn = ttk.Button(reg, text="Bekor qilish", style="Danger.TButton",
                                     command=self.cancel_register)

        emp = self.card(right, "Xodimlar")
        self.emp_tree = ttk.Treeview(emp, columns=("id", "name"), show="headings", height=5)
        self.emp_tree.heading("id", text="ID")
        self.emp_tree.heading("name", text="Ism")
        self.emp_tree.column("id", width=50, anchor="center")
        self.emp_tree.column("name", width=290)
        self.emp_tree.pack(fill="x")
        ttk.Button(emp, text="Tanlangan xodimni o'chirish", style="Danger.TButton",
                   command=self.delete_employee).pack(fill="x", pady=(8, 0))

        log = self.card(right, "Bugungi davomat")
        self.log_tree = ttk.Treeview(log, columns=("time", "name"), show="headings", height=6)
        self.log_tree.heading("time", text="Vaqt")
        self.log_tree.heading("name", text="Ism")
        self.log_tree.column("time", width=90, anchor="center")
        self.log_tree.column("name", width=250)
        self.log_tree.pack(fill="x")

    def card(self, parent, title):
        frame = tk.Frame(parent, bg=CARD, padx=14, pady=12)
        frame.pack(fill="x", pady=(0, 12))
        tk.Label(frame, text=title, bg=CARD, fg=TEXT,
                 font=("Segoe UI", 11, "bold")).pack(anchor="w", pady=(0, 8))
        return frame

    def set_status(self, text, color=MUTED):
        if isinstance(color, tuple):
            color = hexc(color)
        self.status_var.set(text)
        self.status_lbl.configure(fg=color)

    def refresh_lists(self):
        self.emp_tree.delete(*self.emp_tree.get_children())
        for uid, name in sorted(self.employees.items(), key=lambda i: int(i[0])):
            self.emp_tree.insert("", "end", iid=uid, values=(uid, name))
        self.log_tree.delete(*self.log_tree.get_children())
        for r in read_today_log():
            self.log_tree.insert("", "end", values=(r[1], r[3]))

    # ---------- Yuz: topish, sifat, embedding ----------
    def detect(self, frame):
        if self.yunet is not None:
            _, faces = self.yunet.detect(frame)
            if faces is None or len(faces) == 0:
                return None, None
            row = max(faces, key=lambda f: f[2] * f[3])
            x, y, w, h = [int(v) for v in row[:4]]
            return (x, y, w, h), row
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        small = cv2.resize(gray, (0, 0), fx=0.5, fy=0.5)
        faces = self.haar.detectMultiScale(small, 1.2, 5, minSize=(40, 40))
        if len(faces) == 0:
            return None, None
        x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
        return (int(x * 2), int(y * 2), int(w * 2), int(h * 2)), None

    @staticmethod
    def yaw_of(row):
        """Yuzning yon tomonga burilishi (0 = to'g'ri). Landmarklar: ko'z, ko'z, burun, lab, lab."""
        pts = np.asarray(row[4:14], dtype=np.float32).reshape(5, 2)
        eye_d = abs(pts[1][0] - pts[0][0]) + 1e-6
        return float((pts[2][0] - (pts[0][0] + pts[1][0]) / 2) / eye_d)

    def check_quality(self, frame, box, row, max_yaw):
        """(ok, xabar). Yomon sifatli kadrlar ham tanishda, ham ro'yxatga olishda ishlatilmaydi."""
        x, y, w, h = box
        if w < MIN_FACE_PX:
            return False, "Kameraga yaqinroq keling"
        if x < -5 or y < -5 or x + w > VIDEO_W + 5 or y + h > VIDEO_H + 5:
            return False, "Yuz to'liq kadrga sig'sin"
        if row is not None and float(row[14]) < MIN_DET_SCORE:
            return False, "Yuz aniq ko'rinmayapti"
        crop = frame[max(y, 0):max(y + h, 1), max(x, 0):max(x + w, 1)]
        if crop.size == 0:
            return False, "Yuz topilmadi"
        gray = cv2.cvtColor(cv2.resize(crop, (112, 112)), cv2.COLOR_BGR2GRAY)
        bright = float(gray.mean())
        if bright < MIN_BRIGHT:
            return False, "Yorug'lik yetarli emas"
        if bright > MAX_BRIGHT:
            return False, "Yorug'lik juda kuchli"
        if cv2.Laplacian(gray, cv2.CV_64F).var() < MIN_SHARPNESS:
            return False, "Tasvir xira — qimirlamang"
        if row is not None and abs(self.yaw_of(row)) > max_yaw:
            return False, "Kameraga to'g'riroq qarang"
        return True, ""

    def embed(self, frame, box, row, with_flip=False):
        """Normallangan embedding (with_flip=True bo'lsa ko'zgu nusxasi bilan juft)."""
        src = enhance(frame)
        if row is not None:
            aligned = self.sface.alignCrop(src, row)
        else:
            x, y, w, h = box
            crop = src[max(y, 0):max(y + h, 1), max(x, 0):max(x + w, 1)]
            aligned = cv2.resize(crop, (112, 112))
        e = normalize(self.sface.feature(aligned))
        if not with_flip:
            return e
        return e, normalize(self.sface.feature(cv2.flip(aligned, 1)))

    def best_match(self, q):
        """(uid, ball, ikkinchi_ball). Har xodim uchun eng yaxshi TOPK namuna o'rtachasi."""
        scores = []
        for uid, g in self.gallery.items():
            sims = np.sort(g @ q)
            k = min(TOPK, len(sims))
            scores.append((float(sims[-k:].mean()), uid))
        if not scores:
            return None, -1.0, -1.0
        scores.sort(reverse=True)
        second = scores[1][0] if len(scores) > 1 else -1.0
        return scores[0][1], scores[0][0], second

    # ---------- Ro'yxatdan o'tish ----------
    def start_register(self):
        if self.mode == "register":
            return
        name = self.name_var.get().strip()
        if not name:
            messagebox.showwarning("Diqqat", "Avval xodim ismini kiriting.")
            self.entry.focus()
            return
        self.reg_name = name
        self.reg_embs, self.reg_flips = [], []
        self.mode = "register"
        self.reg_btn.state(["disabled"])
        self.progress.configure(value=0)
        self.progress.pack(fill="x", pady=(10, 4))
        self.cancel_btn.pack(fill="x", pady=(4, 0))

    def end_register(self):
        self.mode = "recognize"
        self.reg_embs, self.reg_flips = [], []
        self.progress.pack_forget()
        self.cancel_btn.pack_forget()
        self.reg_btn.state(["!disabled"])
        self.streak = 0
        self.candidate = None
        self.recent.clear()

    def cancel_register(self):
        self.end_register()

    def finish_register(self):
        mean = normalize(np.mean(self.reg_embs, axis=0))
        dup_uid, dup_score, _ = self.best_match(mean)
        if dup_uid is not None and dup_score >= DUP_THRESHOLD:
            if not messagebox.askyesno(
                    "Dublikat?",
                    f"Bu yuz allaqachon '{self.employees[dup_uid]}' sifatida ro'yxatda "
                    f"({dup_score * 100:.0f}% o'xshash).\nBaribir yangi xodim qo'shilsinmi?"):
                self.end_register()
                return
        ids = [int(i) for i in self.employees] or [0]
        uid = str(max(ids) + 1)
        data = np.array(self.reg_embs + self.reg_flips, dtype=np.float32)
        np.save(os.path.join(DATA_DIR, f"{uid}.npy"), data)
        self.employees[uid] = self.reg_name
        save_employees(self.employees)
        self.load_gallery()
        self.name_var.set("")
        self.banner = (f"Saqlandi: {self.reg_name}", GREEN, time.time() + 3)
        self.end_register()
        self.refresh_lists()

    def delete_employee(self):
        sel = self.emp_tree.selection()
        if not sel:
            messagebox.showinfo("Ma'lumot", "Ro'yxatdan xodimni tanlang.")
            return
        uid = sel[0]
        name = self.employees.get(uid, "")
        if messagebox.askyesno("Tasdiqlash", f"{name} o'chirilsinmi?"):
            self.employees.pop(uid, None)
            save_employees(self.employees)
            try:
                os.remove(os.path.join(DATA_DIR, f"{uid}.npy"))
            except OSError:
                pass
            self.load_gallery()
            self.refresh_lists()

    # ---------- Kamera sikli ----------
    def loop(self):
        ok, frame = self.cap.read()
        if ok:
            frame = cv2.flip(frame, 1)
            if frame.shape[1] != VIDEO_W or frame.shape[0] != VIDEO_H:
                frame = cv2.resize(frame, (VIDEO_W, VIDEO_H))
            try:
                box, row = self.detect(frame)
                if self.mode == "register":
                    label, color = self.process_register(frame, box, row)
                else:
                    label, color = self.process_recognize(frame, box, row)
            except Exception as e:
                box, label, color = None, None, BLUE
                self.set_status(f"Xatolik: {e}", RED)
            self.show(frame, box, label, color)

        self.clock_var.set(datetime.now().strftime("%H:%M:%S   %d.%m.%Y"))
        self.root.after(15, self.loop)

    def process_register(self, frame, box, row):
        n = len(self.reg_embs)
        stage = min(n * 3 // SAMPLES, 2)
        hints = ["To'g'ri qarang", "Boshingizni sal chapga buring",
                 "Boshingizni sal o'ngga buring"]
        if box is None:
            self.set_status("Yuz topilmadi — kameraga qarang", AMBER)
            return None, AMBER

        max_yaw = MAX_YAW_FRONT if stage == 0 else MAX_YAW_TURN
        ok, msg = self.check_quality(frame, box, row, max_yaw)
        if not ok:
            self.set_status(msg, AMBER)
            return None, AMBER

        now = time.time()
        if now - self.reg_last < 0.25:
            self.set_status(f"{hints[stage]}  ({n}/{SAMPLES})", BLUE)
            return self.reg_name, BLUE

        emb, flip = self.embed(frame, box, row, with_flip=True)
        limit = DIVERSE_SIM_FRONT if stage == 0 else DIVERSE_SIM_TURN
        if self.reg_embs and float((np.array(self.reg_embs) @ emb).max()) > limit:
            self.set_status(f"{hints[stage]} — biroz boshqacha holatda  ({n}/{SAMPLES})", BLUE)
            return self.reg_name, BLUE

        self.reg_embs.append(emb)
        self.reg_flips.append(flip)
        self.reg_last = now
        self.progress.configure(value=len(self.reg_embs))
        self.set_status(f"{hints[stage]}  ({len(self.reg_embs)}/{SAMPLES})", BLUE)
        if len(self.reg_embs) >= SAMPLES:
            self.finish_register()
        return self.reg_name, BLUE

    def process_recognize(self, frame, box, row):
        if box is None:
            self.streak, self.candidate = 0, None
            self.recent.clear()
            self.set_status("Kameraga qarang", MUTED)
            return None, BLUE
        if not self.gallery:
            self.set_status("Bazada xodim yo'q — avval ro'yxatdan o'ting", AMBER)
            return "Bazada xodim yo'q", AMBER

        ok, msg = self.check_quality(frame, box, row, MAX_YAW_RECOG)
        if not ok:
            self.streak, self.candidate = 0, None
            self.set_status(msg, AMBER)
            return None, AMBER

        emb = self.embed(frame, box, row)
        # boshqa odam kadrga kirgan bo'lsa, eski kadrlarni tashlab yuboramiz
        if self.recent:
            m = normalize(np.mean(self.recent, axis=0))
            if float(emb @ m) < 0.40:
                self.recent.clear()
        self.recent.append(emb)
        avg = normalize(np.mean(self.recent, axis=0))

        uid, sim, second = self.best_match(avg)
        pct = f" {sim * 100:.0f}%" if SHOW_SCORE else ""

        if sim >= MATCH_THRESHOLD and (sim - second) >= MARGIN:
            name = self.employees[uid]
            self.streak = self.streak + 1 if uid == self.candidate else 1
            self.candidate = uid
            if self.streak >= CONFIRM_FRAMES:
                self.check_in(uid, name)
                self.set_status(f"{name} — aniqlandi", GREEN)
                return name + pct, GREEN
            self.set_status("Tekshirilmoqda...", BLUE)
            return name + pct, BLUE

        self.streak, self.candidate = 0, None
        if sim >= MATCH_THRESHOLD:
            self.set_status("Aniq emas — to'g'ri qarab turing", AMBER)
            return "Aniq emas" + pct, AMBER
        self.set_status("Noma'lum yuz — ruxsat yo'q", RED)
        return "Noma'lum" + pct, RED

    def check_in(self, uid, name):
        now = time.time()
        if now - self.last_logged.get(uid, 0) > COOLDOWN_SEC:
            write_log(uid, name)
            self.last_logged[uid] = now
            self.banner = (f"Xush kelibsiz, {name}!", GREEN, now + 3)
            self.refresh_lists()
        elif not self.banner or self.banner[2] < now:
            self.banner = (f"{name} — allaqachon belgilangan", BLUE, now + 2)

    # ---------- Chizish ----------
    def show(self, frame, box, label, color):
        img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        d = ImageDraw.Draw(img)

        if box:
            x, y, w, h = box
            L, t = int(min(w, h) * 0.22), 4
            for cx, cy, dx, dy in ((x, y, 1, 1), (x + w, y, -1, 1),
                                   (x, y + h, 1, -1), (x + w, y + h, -1, -1)):
                d.line([(cx, cy), (cx + dx * L, cy)], fill=color, width=t)
                d.line([(cx, cy), (cx, cy + dy * L)], fill=color, width=t)
            if label:
                tw = d.textlength(label, font=self.font_md)
                tx = max(min(x + (w - tw) / 2, VIDEO_W - tw - 14), 14)
                ty = min(y + h + 10, VIDEO_H - 40)
                d.rounded_rectangle([tx - 12, ty - 4, tx + tw + 12, ty + 32],
                                    radius=10, fill=color)
                d.text((tx, ty), label, font=self.font_md, fill="white")

        if self.banner and self.banner[2] > time.time():
            text, bcolor, _ = self.banner
            d.rectangle([0, 0, VIDEO_W, 48], fill=bcolor)
            tw = d.textlength(text, font=self.font_bn)
            d.text(((VIDEO_W - tw) / 2, 9), text, font=self.font_bn, fill="white")

        self.photo = ImageTk.PhotoImage(img)
        self.video.configure(image=self.photo)

    def on_close(self):
        self.cap.release()
        self.root.destroy()


class FaceRecognitionEngine(App):
    """Headless recognition path that reuses the desktop app's tuned pipeline."""

    def init_models(self):
        """Headless init_models — messagebox yo'q, RuntimeError ishlatiladi."""
        self.sface = None
        self.yunet = None
        self.haar = None

        sface_path = ensure_model(SFACE_FILE, "face_recognition_sface")
        if not sface_path:
            print(f"[XATO] {SFACE_FILE} yuklab bo'lmadi.")
            return False
        try:
            self.sface = cv2.FaceRecognizerSF.create(sface_path, "")
        except Exception as e:
            print(f"[XATO] SFace modelini ochib bo'lmadi: {e}")
            return False

        for fname in YUNET_FILES:
            path = ensure_model(fname, "face_detection_yunet")
            if not path:
                continue
            try:
                det = cv2.FaceDetectorYN.create(path, "", (VIDEO_W, VIDEO_H), 0.8, 0.3, 5000)
                det.setInputSize((VIDEO_W, VIDEO_H))
                det.detect(np.zeros((VIDEO_H, VIDEO_W, 3), np.uint8))
                self.yunet = det
                print("Detektor: YuNet", fname)
                break
            except Exception as e:
                print("YuNet ishlamadi:", fname, e)

        if self.yunet is None:
            self.haar = get_haar()
            if self.haar is None:
                print("[XATO] Yuz topuvchi model yuklanmadi.")
                return False
            print("Detektor: Haar (zaxira)")
        return True

    def __init__(self):
        if not self.init_models():
            raise RuntimeError("Yuzni aniqlash modellari ishga tushmadi.")
        os.makedirs(DATA_DIR, exist_ok=True)
        self.employees = load_employees()
        self.gallery = {}
        self.load_gallery()
        self.candidate = None
        self.streak = 0
        self.recent = deque(maxlen=SMOOTH_FRAMES)

    def reference_embedding(self, image):
        if image is None or image.size == 0:
            raise ValueError("Etalon yuz rasmi o'qilmadi.")
        height, width = image.shape[:2]
        scale = min(VIDEO_W / width, VIDEO_H / height)
        resized = cv2.resize(image, (max(1, int(width * scale)), max(1, int(height * scale))))
        frame = np.full((VIDEO_H, VIDEO_W, 3), 114, dtype=np.uint8)
        x = (VIDEO_W - resized.shape[1]) // 2
        y = (VIDEO_H - resized.shape[0]) // 2
        frame[y:y + resized.shape[0], x:x + resized.shape[1]] = resized
        box, row = self.detect(frame)
        if box is None:
            raise ValueError("Etalon rasmda yuz topilmadi.")
        ok, message = self.check_quality(frame, box, row, MAX_YAW_TURN)
        if not ok:
            raise ValueError(f"Etalon yuz rasmi sifati yetarli emas: {message.lower()}.")
        return self.embed(frame, box, row)

    def recognize_reference(self, frame, reference, employee, state, mirror=False):
        if mirror:
            frame = cv2.flip(frame, 1)
        box, row = self.detect(frame)
        recent = state["recent"]
        if box is None:
            recent.clear()
            state["streak"] = 0
            return {"faceFound": False, "recognized": False, "message": "Kameraga qarang"}

        ok, message = self.check_quality(frame, box, row, MAX_YAW_RECOG)
        if not ok:
            recent.clear()
            state["streak"] = 0
            return {"faceFound": True, "recognized": False, "message": message}

        embedding = self.embed(frame, box, row)
        if recent and float(embedding @ normalize(np.mean(recent, axis=0))) < 0.40:
            recent.clear()
        recent.append(embedding)
        average = normalize(np.mean(recent, axis=0))
        similarity = float(average @ reference)
        if similarity < max(MATCH_THRESHOLD, HRMS_FACE_THRESHOLD):
            state["streak"] = 0
            message = "Yuz etalon rasmga mos kelmadi"
            return {"faceFound": True, "recognized": False, "message": message}

        state["streak"] += 1
        if state["streak"] < CONFIRM_FRAMES:
            return {"faceFound": True, "recognized": False, "message": "Tekshirilmoqda..."}

        confidence = max(0.0, min(1.0, similarity))
        return {
            "faceFound": True,
            "recognized": True,
            "id": str(employee["id"]),
            "name": employee["fullName"],
            "faceConfidence": confidence,
            "faceSimilarity": similarity,
            "message": f"{employee['fullName']} — aniqlandi",
        }

    def recognize(self, frame, mirror=True):
        if mirror:
            frame = cv2.flip(frame, 1)
        box, row = self.detect(frame)
        if box is None:
            self.streak, self.candidate = 0, None
            self.recent.clear()
            return {"faceFound": False, "recognized": False, "message": "Kameraga qarang"}
        if not self.gallery:
            return {"faceFound": True, "recognized": False,
                    "message": "Xodimlarning yuz ma'lumotlari ro'yxatdan o'tkazilmagan"}

        ok, message = self.check_quality(frame, box, row, MAX_YAW_RECOG)
        if not ok:
            self.streak, self.candidate = 0, None
            self.recent.clear()
            return {"faceFound": True, "recognized": False, "message": message}

        embedding = self.embed(frame, box, row)
        if self.recent:
            average = normalize(np.mean(self.recent, axis=0))
            if float(embedding @ average) < 0.40:
                self.recent.clear()
        self.recent.append(embedding)
        average = normalize(np.mean(self.recent, axis=0))
        uid, score, second = self.best_match(average)
        if score >= MATCH_THRESHOLD and score - second >= MARGIN:
            self.streak = self.streak + 1 if uid == self.candidate else 1
            self.candidate = uid
            if self.streak >= CONFIRM_FRAMES:
                return {"faceFound": True, "recognized": True, "id": uid,
                        "name": self.employees[uid], "message": f"{self.employees[uid]} — aniqlandi"}
            return {"faceFound": True, "recognized": False, "message": "Tekshirilmoqda..."}

        self.streak, self.candidate = 0, None
        message = "Aniq emas — to'g'ri qarab turing" if score >= MATCH_THRESHOLD else "Noma'lum yuz"
        return {"faceFound": True, "recognized": False, "message": message}


if __name__ == "__main__":
    import sys

    if "--desktop" in sys.argv:
        root = tk.Tk()
        app = App(root)
        if app.ok:
            root.mainloop()
    else:
        from server import run_server

        run_server(FaceRecognitionEngine)