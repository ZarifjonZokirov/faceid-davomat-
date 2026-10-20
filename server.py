import base64
import csv
import ipaddress
import json
import math
import os
import secrets
import socket
import threading
import time
from collections import deque
from http.cookies import SimpleCookie
from datetime import datetime
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from urllib.error import HTTPError, URLError
from urllib.parse import parse_qs, urlencode, urlparse
from urllib.request import HTTPRedirectHandler, Request, build_opener

import cv2
import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
LOG_FILE = os.path.join(BASE_DIR, "attendance.csv")
SESSION_FILE = os.path.join(BASE_DIR, "sessions.json")
HRMS_BASE_URL = os.environ.get("HRMS_BASE_URL", "https://hrms-toyona.vercel.app").rstrip("/")
SESSION_COOKIE = "toyona_attendance_session"
SESSION_TTL = 24 * 60 * 60
FACE_TOKEN_TTL = 60
MAX_API_RESPONSE = 7_000_000
MAX_REFERENCE_IMAGE = 5_000_000


def load_persisted_sessions():
    """Fayldan saqlangan sessiyalarni yuklash (serverlar qayta ishga tushganda ham saqlandi)."""
    try:
        if not os.path.exists(SESSION_FILE):
            return {}
        with open(SESSION_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        now = time.time()
        # Muddati o'tgan sessiyalarni o'chirish
        valid = {k: v for k, v in data.items() if v.get("expiresAt", 0) > now}
        return valid
    except Exception:
        return {}


def persist_sessions(sessions: dict):
    """Barcha faol sessiyalarni faylga yozish (embedding saqlanmaydi — xavfsizlik uchun)."""
    try:
        now = time.time()
        serializable = {}
        for sid, s in sessions.items():
            if s.get("expiresAt", 0) <= now:
                continue
            serializable[sid] = {
                "sessionToken": s.get("sessionToken", ""),
                "employee": s.get("employee", {}),
                "branchGeozone": s.get("branchGeozone"),
                "expiresAt": s.get("expiresAt", 0),
                "_cached_photo": s.get("_cached_photo"),
            }
        with open(SESSION_FILE, "w", encoding="utf-8") as f:
            json.dump(serializable, f, ensure_ascii=False)
    except Exception as e:
        print(f"[Session] Sessiyalarni saqlashda xatolik: {e}")


def send_json(handler, status, payload, headers=None):
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    handler.send_response(status)
    handler.send_header("Content-Type", "application/json; charset=utf-8")
    handler.send_header("Content-Length", str(len(body)))
    handler.send_header("Cache-Control", "no-store")
    for name, value in (headers or {}).items():
        handler.send_header(name, value)
    handler.end_headers()
    handler.wfile.write(body)


def read_rows():
    if not os.path.exists(LOG_FILE):
        return []
    with open(LOG_FILE, "r", newline="", encoding="utf-8-sig") as file:
        return list(csv.reader(file))[1:]


class NoRedirectHandler(HTTPRedirectHandler):
    def redirect_request(self, _req, _fp, _code, _msg, _headers, _newurl):
        return None


def hrms_request(path, method="GET", payload=None, session_token=None):
    body = json.dumps(payload).encode("utf-8") if payload is not None else None
    headers = {"Accept": "application/json", "Content-Type": "application/json"}
    if session_token:
        headers["Authorization"] = f"Bearer {session_token}"
    request = Request(HRMS_BASE_URL + path, data=body, headers=headers, method=method)
    try:
        response = build_opener(NoRedirectHandler).open(request, timeout=15)
    except HTTPError as error:
        response = error
    except (URLError, TimeoutError, OSError):
        return 502, {"success": False, "code": "HRMS_UNAVAILABLE",
                     "message": "HRMS serveri bilan bog'lanib bo'lmadi."}

    with response:
        raw = response.read(MAX_API_RESPONSE + 1)
        content_type = response.headers.get("Content-Type", "").lower()
        if len(raw) > MAX_API_RESPONSE:
            return 502, {"success": False, "code": "HRMS_RESPONSE_TOO_LARGE",
                         "message": "HRMS javobi juda katta."}
        if "application/json" not in content_type:
            return 502, {"success": False, "code": "HRMS_NON_JSON_RESPONSE",
                         "message": "HRMS endpoint JSON qaytarmadi."}
        try:
            data = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            return 502, {"success": False, "code": "HRMS_INVALID_JSON",
                         "message": "HRMS javobidagi JSON noto'g'ri."}
        return response.status, data


def fetch_reference_image(photo_url):
    if not isinstance(photo_url, str):
        raise ValueError("HRMS xodimning etalon rasmi manzilini bermadi.")
    if photo_url.startswith("data:image/"):
        header, separator, encoded = photo_url.partition(",")
        if not separator or ";base64" not in header.lower():
            raise ValueError("Etalon rasm data URL formati noto'g'ri.")
        try:
            data = base64.b64decode(encoded, validate=True)
        except (ValueError, TypeError) as error:
            raise ValueError("Etalon rasm base64 formati noto'g'ri.") from error
    else:
        parsed = urlparse(photo_url)
        if parsed.scheme not in ("https", "http") or not parsed.hostname:
            raise ValueError("Etalon rasm HTTP yoki HTTPS manzilda bo'lishi kerak.")
        is_local = parsed.hostname in ("127.0.0.1", "localhost")
        if not is_local:
            try:
                addresses = socket.getaddrinfo(parsed.hostname, parsed.port or 443, type=socket.SOCK_STREAM)
                if not addresses or any(not ipaddress.ip_address(item[4][0]).is_global for item in addresses):
                    raise ValueError("Etalon rasm manzili ruxsat etilgan tashqi host emas.")
            except (OSError, ValueError) as error:
                raise ValueError("Etalon rasm hostini tekshirib bo'lmadi.") from error

        request = Request(photo_url, headers={"Accept": "image/*", "User-Agent": "Toyona-FaceID/1.0"})
        try:
            response = build_opener(NoRedirectHandler).open(request, timeout=12)
        except (HTTPError, URLError, TimeoutError, OSError) as error:
            raise ValueError("Xodimning etalon rasmini yuklab bo'lmadi.") from error
        with response:
            content_type = response.headers.get("Content-Type", "").lower()
            if not content_type.startswith("image/"):
                raise ValueError("Xodimning photoUrl manzili rasm qaytarmadi.")
            data = response.read(MAX_REFERENCE_IMAGE + 1)
    if len(data) > MAX_REFERENCE_IMAGE:
        raise ValueError("Etalon rasm hajmi 5 MB dan oshmasligi kerak.")
    image = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("Etalon yuz rasmini o'qib bo'lmadi.")
    return image


class AttendanceHandler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=BASE_DIR, **kwargs)

    def do_GET(self):
        parsed = urlparse(self.path)
        if parsed.path == "/api/hrms/session":
            current = self.get_hrms_session()
            if current is None:
                send_json(self, 401, {"success": False, "message": "Sessiya tugagan."})
                return
            session_id, session = current
            # HAR REFRESH BO'LGANDA: HRMS'dan yangi ma'lumotlar (filial, koordinatalar, rasm) avtomatik yangilanadi
            try:
                emp_id = session["employee"]["id"]
                token = session["sessionToken"]
                
                # 1. Filiallar ro'yxati va koordinatalarini yangilab olamiz
                z_status, zones = hrms_request("/api/geozones", session_token=token)
                latest_zones = zones if (z_status == 200 and isinstance(zones, list)) else []

                # 2. Xodimning joriy ma'lumotlarini (filiali o'zgargan bo'lsa ham) yangilaymiz
                e_status, e_data = hrms_request("/api/attendance/employees", session_token=token)
                emp_list = e_data if isinstance(e_data, list) else (e_data.get("employees", []) if isinstance(e_data, dict) else [])
                current_emp = next((item for item in emp_list if str(item.get("id")) == str(emp_id)), None)
                if current_emp:
                    session["employee"].update(current_emp)

                # 3. Geozonani xodimning yangilangan filialiga qarab yangilaymiz
                emp_branch = session["employee"].get("branch")
                if latest_zones and emp_branch:
                    matched_zone = next((z for z in latest_zones if (z.get("name") or "").strip().lower() == emp_branch.strip().lower() or str(z.get("id")) == str(session["employee"].get("branchId"))), None)
                    if matched_zone:
                        session["branchGeozone"] = matched_zone

                # 4. Rasm yangilangan bo'lsa yangi embedding hisoblaymiz
                new_photo = session["employee"].get("photoUrl") or session["employee"].get("rasmUrl")
                if not new_photo:
                    st_code, p_data = hrms_request("/api/attendance/employee-photo?" + urlencode({"employeeId": str(emp_id)}), session_token=token)
                    if st_code == 200 and isinstance(p_data, dict):
                        new_photo = p_data.get("photoUrl")
                
                if new_photo and (new_photo != session.get("_cached_photo") or "referenceEmbedding" not in session):
                    print(f"[Auto-Refresh] Yangi rasm yuklandi: {new_photo[:40]}...")
                    new_img = fetch_reference_image(new_photo)
                    new_emb = self.server.recognition_engine.reference_embedding(new_img)
                    session["employee"]["photoUrl"] = new_photo
                    session["referenceEmbedding"] = new_emb
                    session["_cached_photo"] = new_photo

                # Yangilangan sessiyani xotirada va faylda saqlaymiz
                with self.server.session_lock:
                    self.server.hrms_sessions[session_id] = session
                    persist_sessions(self.server.hrms_sessions)

            except Exception as e:
                print(f"[Auto-Refresh] Yangilashda xatolik: {e}")

            send_json(self, 200, {"success": True, "employee": session["employee"],
                                 "branchGeozone": session["branchGeozone"]})
            return
        if parsed.path == "/api/hrms/geozones":
            current = self.get_hrms_session()
            if current is None:
                send_json(self, 401, {"success": False, "message": "Avval tizimga kiring."})
                return
            _, session = current
            status, data = hrms_request("/api/geozones", session_token=session["sessionToken"])
            send_json(self, status, data)
            return
        if parsed.path == "/api/hrms/penalty-settings":
            current = self.get_hrms_session()
            token = current[1]["sessionToken"] if current else None
            status, data = hrms_request("/api/attendance/penalty-settings", session_token=token)
            send_json(self, status, data)
            return
        if parsed.path == "/api/hrms/history":
            current = self.get_hrms_session()
            if current is None:
                send_json(self, 401, {"success": False, "message": "Avval tizimga kiring."})
                return
            _, session = current
            params = parse_qs(parsed.query)
            month = params.get("month", [datetime.now().strftime("%Y-%m")])[0]
            year = params.get("year", [""])[0]
            q_dict = {"employeeId": session["employee"]["id"]}
            if month:
                q_dict["month"] = month
            if year:
                q_dict["year"] = year
            query = urlencode(q_dict)
            status, data = hrms_request("/api/attendance/history?" + query,
                                        session_token=session["sessionToken"])
            send_json(self, status, data)
            return
        if parsed.path == "/api/attendance":
            month = parse_qs(parsed.query).get("month", [datetime.now().strftime("%Y-%m")])[0]
            employee_id = parse_qs(parsed.query).get("id", [""])[0]
            days = {}
            for row in read_rows():
                if len(row) < 4 or row[2] != employee_id or not row[0].startswith(month):
                    continue
                day = row[0][8:10]
                record = days.setdefault(day, {"day": day, "start": "--:--", "end": "--:--", "missed": False})
                if record["start"] == "--:--":
                    record["start"] = row[1][:5]
                else:
                    record["end"] = row[1][:5]
            send_json(self, 200, {"records": list(days.values())})
            return
        if parsed.path.startswith("/api/"):
            send_json(self, 404, {"success": False, "message": "API endpoint topilmadi."})
            return
        super().do_GET()

    def do_POST(self):
        parsed = urlparse(self.path)
        try:
            length = int(self.headers.get("Content-Length", "0"))
            if length <= 0 or length > 10_000_000:
                send_json(self, 400, {"error": "So'rov hajmi noto'g'ri."})
                return
            payload = json.loads(self.rfile.read(length))
            if not isinstance(payload, dict):
                send_json(self, 400, {"error": "JSON obyekt yuborilishi kerak."})
                return
            if parsed.path == "/api/hrms/login":
                self.hrms_login(payload)
            elif parsed.path == "/api/hrms/logout":
                self.hrms_logout()
            elif parsed.path == "/api/hrms/check-in":
                self.hrms_check_in(payload)
            elif parsed.path == "/api/scan":
                self.recognize_face(payload)
            else:
                send_json(self, 404, {"error": "API topilmadi."})
        except (ValueError, json.JSONDecodeError):
            send_json(self, 400, {"error": "So'rovni o'qib bo'lmadi."})

    def get_hrms_session(self):
        cookie = SimpleCookie()
        try:
            cookie.load(self.headers.get("Cookie", ""))
        except Exception:
            return None
        morsel = cookie.get(SESSION_COOKIE)
        if morsel is None:
            return None
        session_id = morsel.value
        with self.server.session_lock:
            session = self.server.hrms_sessions.get(session_id)
            if session is not None and session["expiresAt"] <= time.time():
                self.server.hrms_sessions.pop(session_id, None)
                session = None
        return (session_id, session) if session is not None else None

    def hrms_login(self, payload):
        username = payload.get("login", payload.get("username"))
        password = payload.get("password")
        if not isinstance(username, str) or not username.strip() or not isinstance(password, str) or not password:
            send_json(self, 400, {"success": False, "message": "Login va parolni kiriting."})
            return
        status, data = hrms_request("/api/attendance/login", "POST",
                        {"username": username.strip(), "password": password})
        if status != 200 or not isinstance(data, dict) or data.get("success") is not True:
            message = data.get("message", "HRMS login amalga oshmadi.") if isinstance(data, dict) else "HRMS javobi noto'g'ri."
            send_json(self, status if status >= 400 else 401,
                      {"success": False, "message": message, "code": data.get("code") if isinstance(data, dict) else None})
            return

        employee = data.get("employee")
        remote_token = data.get("sessionToken")
        if not isinstance(employee, dict) or not employee.get("id") or not employee.get("fullName") or not remote_token:
            send_json(self, 502, {"success": False, "message": "HRMS login javobida xodim yoki sessiya ma'lumoti yo'q."})
            return
        employee = dict(employee)
        employee.setdefault("username", username.strip())

        employee_status, employee_data = hrms_request(
            "/api/attendance/employees", session_token=remote_token)
        if employee_status == 200:
            if isinstance(employee_data, list):
                employees = employee_data
            elif isinstance(employee_data, dict):
                employees = employee_data.get("employees", [])
            else:
                employees = []
            if isinstance(employees, list):
                current = next((item for item in employees
                                if isinstance(item, dict) and str(item.get("id")) == str(employee["id"])), None)
                if current:
                    employee["isVacationActive"] = bool(current.get("isVacationActive"))
                    employee["isTripActive"] = bool(current.get("isTripActive"))

        photo_url = employee.get("photoUrl") or employee.get("rasmUrl")
        if not photo_url:
            photo_status, photo_data = hrms_request(
                "/api/attendance/employee-photo?" + urlencode({"employeeId": str(employee["id"])}),
                session_token=remote_token)
            if photo_status == 200 and isinstance(photo_data, dict):
                photo_url = photo_data.get("photoUrl")
                employee["photoUrl"] = photo_url
        try:
            reference_image = fetch_reference_image(photo_url)
            reference_embedding = self.server.recognition_engine.reference_embedding(reference_image)
        except (ValueError, KeyError) as error:
            send_json(self, 422, {"success": False, "message": str(error)})
            return

        branch_geozone = data.get("branchGeozone")
        if not branch_geozone:
            zone_status, zones = hrms_request("/api/geozones", session_token=remote_token)
            if zone_status == 200 and isinstance(zones, list):
                branch_geozone = next((zone for zone in zones if zone.get("name") == employee.get("branch")), None)

        local_session_id = secrets.token_urlsafe(32)
        session = {
            "sessionToken": remote_token,
            "employee": employee,
            "branchGeozone": branch_geozone,
            "referenceEmbedding": reference_embedding,
            "recognitionState": {"recent": deque(maxlen=5), "streak": 0},
            "expiresAt": time.time() + SESSION_TTL,
        }
        with self.server.session_lock:
            self.server.hrms_sessions[local_session_id] = session
            persist_sessions(self.server.hrms_sessions)
        cookie = f"{SESSION_COOKIE}={local_session_id}; HttpOnly; SameSite=Strict; Path=/; Max-Age={SESSION_TTL}"
        send_json(self, 200, {"success": True, "employee": employee, "branchGeozone": branch_geozone},
                  {"Set-Cookie": cookie})

    def hrms_logout(self):
        current = self.get_hrms_session()
        if current:
            with self.server.session_lock:
                self.server.hrms_sessions.pop(current[0], None)
                persist_sessions(self.server.hrms_sessions)
        cookie = f"{SESSION_COOKIE}=; HttpOnly; SameSite=Strict; Path=/; Max-Age=0"
        send_json(self, 200, {"success": True}, {"Set-Cookie": cookie})

    def recognize_face(self, payload):
        current = self.get_hrms_session()
        if current is None:
            send_json(self, 401, {"success": False, "message": "Avval HRMS hisobiga kiring."})
            return
        session_id, session = current
        image_data = payload.get("image", "")
        if not isinstance(image_data, str):
            send_json(self, 400, {"success": False, "message": "Kamera tasviri noto'g'ri."})
            return
        if "," in image_data:
            image_data = image_data.split(",", 1)[1]
        try:
            encoded = base64.b64decode(image_data, validate=True)
        except (ValueError, TypeError):
            send_json(self, 400, {"success": False, "message": "Kamera tasviri noto'g'ri."})
            return
        image = cv2.imdecode(np.frombuffer(encoded, np.uint8), cv2.IMREAD_COLOR)
        if image is None:
            send_json(self, 400, {"error": "Kamera tasvirini o'qib bo'lmadi."})
            return
        image = cv2.resize(image, (640, 480))
        with self.server.recognition_lock:
            result = self.server.recognition_engine.recognize_reference(
                image, session["referenceEmbedding"], session["employee"],
                session["recognitionState"], mirror=False)
            if result["recognized"]:
                now = time.monotonic()
                for token, record in list(self.server.face_tokens.items()):
                    if record["expiresAt"] <= now:
                        self.server.face_tokens.pop(token, None)
                face_token = secrets.token_urlsafe(32)
                self.server.face_tokens[face_token] = {
                    "sessionId": session_id,
                    "employeeId": str(session["employee"]["id"]),
                    "capturedPhoto": payload.get("image", ""),
                    "faceConfidence": result["faceConfidence"],
                    "expiresAt": now + FACE_TOKEN_TTL,
                }
                result["token"] = face_token
        send_json(self, 200, result)

    def hrms_check_in(self, payload):
        current = self.get_hrms_session()
        if current is None:
            send_json(self, 401, {"success": False, "message": "Avval HRMS hisobiga kiring."})
            return
        session_id, session = current
        token = payload.get("faceToken", "")
        if not isinstance(token, str) or not token:
            send_json(self, 403, {"success": False, "message": "Yuz tasdig'i tokeni noto'g'ri."})
            return
        with self.server.recognition_lock:
            verified = self.server.face_tokens.pop(token, None)
        if (verified is None or verified["expiresAt"] < time.monotonic()
                or verified["sessionId"] != session_id
                or verified["employeeId"] != str(session["employee"]["id"])):
            send_json(self, 403, {"success": False, "message": "Yuz tasdig'i topilmadi yoki muddati o'tdi."})
            return
        action = payload.get("action")
        if action not in ("keldim", "ketdim"):
            send_json(self, 400, {"success": False, "message": "Davomat turi noto'g'ri."})
            return
        try:
            latitude = float(payload.get("userLat"))
            longitude = float(payload.get("userLon"))
        except (TypeError, ValueError):
            send_json(self, 400, {"success": False, "message": "GPS koordinatalari noto'g'ri."})
            return
        if (not math.isfinite(latitude) or not math.isfinite(longitude)
                or not -90 <= latitude <= 90 or not -180 <= longitude <= 180):
            send_json(self, 400, {"success": False, "message": "GPS koordinatalari noto'g'ri."})
            return
        # Geozonani har ehtimolga qarshi oxirgi HRMS holatidan yangilab olamiz
        branch_geozone = session.get("branchGeozone")
        try:
            z_status, zones = hrms_request("/api/geozones", session_token=session.get("sessionToken"))
            if z_status == 200 and isinstance(zones, list):
                emp_branch = session["employee"].get("branch")
                matched_zone = next((z for z in zones if (z.get("name") or "").strip().lower() == (emp_branch or "").strip().lower() or str(z.get("id")) == str(session["employee"].get("branchId"))), None)
                if matched_zone:
                    branch_geozone = matched_zone
                    session["branchGeozone"] = matched_zone
        except Exception:
            pass

        branch_id = branch_geozone.get("id") if isinstance(branch_geozone, dict) else None
        if not branch_id:
            send_json(self, 422, {"success": False, "message": "Xodim uchun filial geozonasi topilmadi."})
            return
        body = {
            "employeeId": str(session["employee"]["id"]),
            "username": session["employee"]["username"],
            "action": action,
            "branchId": str(branch_id),
            "branchName": branch_geozone.get("name"),
            "userLat": latitude,
            "userLon": longitude,
            "faceVerified": True,
            "faceConfidence": verified["faceConfidence"],
            "capturedPhoto": verified["capturedPhoto"],
        }
        status, result = hrms_request("/api/attendance/check-in", "POST", body,
                                      session_token=session["sessionToken"])
        send_json(self, status, result)


class AttendanceServer(ThreadingHTTPServer):
    def __init__(self, address, handler, engine):
        super().__init__(address, handler)
        self.recognition_engine = engine
        self.recognition_lock = threading.Lock()
        self.session_lock = threading.Lock()
        # Fayldan saqlangan sessiyalarni yuklash
        persisted = load_persisted_sessions()
        self.hrms_sessions = persisted
        if persisted:
            print(f"[Session] {len(persisted)} ta saqlangan sessiya yuklandi.")
        self.face_tokens = {}


def run_server(engine_class):
    engine = engine_class()
    port = int(os.environ.get("PORT", 8000))
    host = os.environ.get("HOST", "0.0.0.0")
    server = AttendanceServer((host, port), AttendanceHandler, engine)
    print("Davomat ilovasi: http://127.0.0.1:8000")
    print(f"Yuz tanish: main.py | Xodimlar bazada: {len(engine.employees)}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nServer to'xtatildi.")
        with server.session_lock:
            persist_sessions(server.hrms_sessions)
        server.server_close()


if __name__ == "__main__":
    from main import FaceRecognitionEngine

    run_server(FaceRecognitionEngine)