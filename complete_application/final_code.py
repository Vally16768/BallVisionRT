import cv2
import numpy as np
from sklearn.cluster import KMeans
import csv
import math
import os

# ==========================
# CONFIG
# ==========================
VIDEO_PATH = "rgb.avi"  # fișierul video de intrare

OUTPUT_DIR = "output_dotlumen"
os.makedirs(OUTPUT_DIR, exist_ok=True)

OUTPUT_FIRST_FRAME = os.path.join(OUTPUT_DIR, "frame0.png")
OUTPUT_FLOOR_MASK = os.path.join(OUTPUT_DIR, "floor_mask.png")
OUTPUT_FLOOR_OVERLAY = os.path.join(OUTPUT_DIR, "floor_overlay.png")
OUTPUT_FLOOR_TOPVIEW = os.path.join(OUTPUT_DIR, "floor_topview.png")

OUTPUT_DET_VIDEO = os.path.join(OUTPUT_DIR, "ball_detection.avi")
OUTPUT_TRAJ_VIDEO = os.path.join(OUTPUT_DIR, "ball_trajectory.avi")
OUTPUT_TOPVIEW_VIDEO = os.path.join(OUTPUT_DIR, "topview_map.avi")
OUTPUT_TOPVIEW_IMAGE = os.path.join(OUTPUT_DIR, "topview_final.png")
OUTPUT_CSV = os.path.join(OUTPUT_DIR, "ball_3d_positions_and_velocities.csv")

# RealSense D435i RGB approximate HFOV (folosit pentru focala în px)
D435I_RGB_HFOV_DEG = 69.0

# Minge de fotbal ~22cm
BALL_DIAMETER_M = 0.22
BALL_RADIUS_M = BALL_DIAMETER_M / 2.0

# Dimensiunea top-view (world map 2D)
TOP_W = 800
TOP_H = 600


# ==========================
# Utilitare – podea
# ==========================
def read_first_frame(video_path):
    """
    WHAT: Citește primul frame din video.
    WHY: Folosim frame0 ca referință pentru:
         - segmentarea podelei (world reference)
         - definirea homografiei către top-view
         - template pentru tracking-ul camerei.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Nu pot deschide video: {video_path}")

    ok, frame = cap.read()
    cap.release()

    if not ok or frame is None:
        raise RuntimeError("Nu am reușit să citesc primul frame din video.")

    return frame


def segment_floor_kmeans(bgr_img, k=4):
    """
    WHAT:
      Segmentează podeaua din frame0 pe baza culorii (KMeans în spațiul BGR + heuristici).
    WHY:
      Avem nevoie de o mască de podea:
        - ca să considerăm doar zona de interes (ground plane),
        - ca să definim bounding box-ul folosit la tracking-ul camerei,
        - ca să filtrăm detecția mingii (mingea trebuie să fie pe podea).
    """
    h, w = bgr_img.shape[:2]

    # micșorăm imaginea pentru KMeans (mai rapid, suficient pentru segmentare)
    scale = 0.25
    small = cv2.resize(bgr_img, (int(w * scale), int(h * scale)))
    sh, sw = small.shape[:2]

    data = small.reshape(-1, 3).astype(np.float32)

    kmeans = KMeans(n_clusters=k, n_init=5, random_state=0)
    kmeans.fit(data)
    labels = kmeans.labels_.reshape(sh, sw)

    # trecem în LAB ca să avem luminozitatea (L)
    lab_small = cv2.cvtColor(small, cv2.COLOR_BGR2LAB)
    L = lab_small[:, :, 0]

    best_score = -1.0
    best_idx = 0

    # alegem clusterul cel mai "probabil podea" (jos, mare, nu foarte luminos)
    for i in range(k):
        ys, xs = np.where(labels == i)
        if ys.size == 0:
            continue

        frac_bottom = ys.mean() / sh           # 0 sus, 1 jos
        area_frac = ys.size / float(sh * sw)   # cât din imagine ocupă
        mean_L = L[ys, xs].mean() / 255.0      # luminozitate normalizată

        score = frac_bottom + 0.5 * area_frac - 0.3 * mean_L

        if score > best_score:
            best_score = score
            best_idx = i

    # mască pentru clusterul ales (în rezoluție mică)
    floor_mask_small = (labels == best_idx).astype(np.uint8) * 255

    # upsampling la rezoluția originală
    floor_mask = cv2.resize(
        floor_mask_small, (w, h), interpolation=cv2.INTER_NEAREST
    )

    # curățare zgomot cu morfologie
    kernel = np.ones((25, 25), np.uint8)
    floor_mask = cv2.morphologyEx(floor_mask, cv2.MORPH_CLOSE, kernel)
    floor_mask = cv2.morphologyEx(floor_mask, cv2.MORPH_OPEN, kernel)

    # păstrăm doar componenta conectată cea mai mare (probabil podeaua completă)
    contours, _ = cv2.findContours(
        floor_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    if not contours:
        raise RuntimeError("Nu am putut găsi niciun contur pentru podea.")

    largest = max(contours, key=cv2.contourArea)
    mask_clean = np.zeros_like(floor_mask)
    cv2.drawContours(mask_clean, [largest], -1, 255, thickness=-1)

    return mask_clean


def create_overlay(frame, floor_mask):
    """
    WHAT:
      Suprapune podeaua în verde semi-transparent peste frame.
    WHY:
      Verificare vizuală rapidă că segmentarea podelei este OK.
    """
    overlay = frame.copy()
    green = np.zeros_like(frame)
    green[:, :, 1] = 255  # canal G

    alpha = 0.3
    mask_3ch = cv2.merge([floor_mask, floor_mask, floor_mask]) / 255.0
    overlay = (frame * (1 - alpha * mask_3ch) + green * (alpha * mask_3ch)).astype(
        np.uint8
    )
    return overlay


def compute_topview_and_H(frame, floor_mask):
    """
    WHAT:
      Definește o homografie (frame0 -> top-view) folosind 4 puncte pe podea
      (alegi manual colțuri aproximative) și proiectează masca de podea într-o hartă top-view.
    WHY:
      Avem nevoie de un sistem 2D "world" (bird's-eye), în care vom
      plasa atât mingea, cât și camera, independent de mișcarea camerei.
    """
    h, w = frame.shape[:2]

    # aceste puncte trebuie măsurate / ajustate pentru scena ta reală
    src = np.float32([
        [100, 300],        # stânga-sus
        [w - 100, 300],    # dreapta-sus
        [w - 50, h - 50],  # dreapta-jos
        [50, h - 50],      # stânga-jos
    ])

    top_w = TOP_W
    top_h = TOP_H
    dst = np.float32([
        [0, 0],
        [top_w - 1, 0],
        [top_w - 1, top_h - 1],
        [0, top_h - 1],
    ])

    # homografie frame0 -> top-view
    H = cv2.getPerspectiveTransform(src, dst)

    # proiectăm masca de podea în top-view
    topview_mask = cv2.warpPerspective(
        floor_mask, H, (top_w, top_h), flags=cv2.INTER_NEAREST
    )

    return H, topview_mask


# ==========================
# Helpers RealSense + minge
# ==========================
def estimate_focal_length_pixels(width, hfov_deg):
    """
    WHAT:
      Estimează focala în pixeli din lățimea imaginii și HFOV.
    WHY:
      Avem nevoie de focala în pixeli pentru modelul pinhole:
        Z = f * R / r_px.
    """
    hfov_rad = math.radians(hfov_deg)
    f = (width / 2.0) / math.tan(hfov_rad / 2.0)
    return f


def detect_red_ball_on_floor(frame, floor_mask, prev_center=None, max_shift=150):
    """
    WHAT:
      Detectează o minge roșie în spațiul HSV, restricționată DOAR la zona de podea.
      Întoarce (x, y, r) sau None.
    WHY:
      - ne concentrăm pe culoarea mingii (roșu),
      - excludem toate zonele non-podea (haine, pereți, cer etc.),
      - folosim contururi și cerc minim înconjurător pentru robusteză.
    """
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # roșu în HSV e împărțit în două intervale
    lower_red1 = np.array([0, 100, 80], dtype=np.uint8)
    upper_red1 = np.array([10, 255, 255], dtype=np.uint8)

    lower_red2 = np.array([170, 100, 80], dtype=np.uint8)
    upper_red2 = np.array([180, 255, 255], dtype=np.uint8)

    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    mask_red = cv2.bitwise_or(mask1, mask2)

    # restricționăm roșul DOAR la podea
    mask_red = cv2.bitwise_and(mask_red, mask_red, mask=floor_mask)

    # curățăm zgomotul
    kernel = np.ones((5, 5), np.uint8)
    mask_red = cv2.morphologyEx(mask_red, cv2.MORPH_OPEN, kernel, iterations=2)
    mask_red = cv2.morphologyEx(mask_red, cv2.MORPH_CLOSE, kernel, iterations=2)

    contours, _ = cv2.findContours(mask_red, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    candidate_circles = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 50:
            continue
        (x, y), r = cv2.minEnclosingCircle(cnt)
        x = int(x)
        y = int(y)
        r = int(r)
        if r < 5:
            continue
        candidate_circles.append((x, y, r, area))

    if not candidate_circles:
        return None

    # dacă nu avem istoric, alegem cel mai mare contur
    if prev_center is None:
        candidate_circles.sort(key=lambda c: -c[3])  # sort desc by area
        x, y, r, _ = candidate_circles[0]
        return (x, y, r)

    # altfel, preferăm cercul cel mai apropiat de poziția anterioară (tracking temporal)
    px, py = prev_center
    best = None
    best_dist = 1e9
    for x, y, r, area in candidate_circles:
        d = math.hypot(x - px, y - py)
        if d < best_dist and d < max_shift:
            best_dist = d
            best = (x, y, r)

    if best is None:
        candidate_circles.sort(key=lambda c: -c[3])
        x, y, r, _ = candidate_circles[0]
        return (x, y, r)

    return best


def estimate_ball_3d(u, v, r_px, f_px, cx, cy):
    """
    WHAT:
      Estimează poziția 3D a mingii (X,Y,Z) + distanța, în metri, în sistemul camerei.
    WHY:
      Folosim un model pinhole simplu:
        Z = f * R / r_px,
        X = (u - cx) * Z / f,
        Y = (v - cy) * Z / f.
      Unde:
        - R = raza reală a mingii,
        - r_px = raza în pixeli,
        - f = focala în pixeli.
    """
    if r_px <= 0:
        return None

    Z = f_px * BALL_RADIUS_M / float(r_px)
    X = (u - cx) * Z / f_px
    Y = (v - cy) * Z / f_px
    dist = math.sqrt(X * X + Y * Y + Z * Z)
    return X, Y, Z, dist


# ==========================
# MAIN PIPELINE
# ==========================
def main():
    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        raise RuntimeError(f"Nu pot deschide video: {VIDEO_PATH}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 30.0  # fallback

    # --- 1) FRAME0: referință pentru podea + homografie + tracking cameră ---
    frame0 = read_first_frame(VIDEO_PATH)
    h, w = frame0.shape[:2]
    print(f"[INFO] Rezoluție video: {w}x{h}, FPS={fps:.2f}")

    cv2.imwrite(OUTPUT_FIRST_FRAME, frame0)
    print(f"[OK] Primul frame salvat ca {OUTPUT_FIRST_FRAME}")

    # Segmentarea podelei (în frame0)
    floor_mask0 = segment_floor_kmeans(frame0)
    cv2.imwrite(OUTPUT_FLOOR_MASK, floor_mask0)
    print(f"[OK] Mască podea salvată ca {OUTPUT_FLOOR_MASK}")

    # Overlay pentru debug
    overlay0 = create_overlay(frame0, floor_mask0)
    cv2.imwrite(OUTPUT_FLOOR_OVERLAY, overlay0)
    print(f"[OK] Overlay salvat ca {OUTPUT_FLOOR_OVERLAY}")

    # Homografie frame0 -> top-view + hartă podea
    H0_to_top, floor_topview0 = compute_topview_and_H(frame0, floor_mask0)
    cv2.imwrite(OUTPUT_FLOOR_TOPVIEW, floor_topview0)
    print(f"[OK] Hartă 2D top-view podea salvată ca {OUTPUT_FLOOR_TOPVIEW}")

    # bounding box al podelei (în frame0) – template pentru tracking cameră
    ys, xs = np.where(floor_mask0 > 0)
    y_min, y_max = ys.min(), ys.max()
    x_min, x_max = xs.min(), xs.max()

    gray0 = cv2.cvtColor(frame0, cv2.COLOR_BGR2GRAY)
    floor_template = gray0[y_min:y_max + 1, x_min:x_max + 1]
    th, tw = floor_template.shape[:2]
    print(f"[INFO] Template podea: {tw}x{th} px")

    # top-view: pornim de la podea ca fundal (world map fix)
    topview_accum = cv2.cvtColor(floor_topview0, cv2.COLOR_GRAY2BGR)

    # Estimare focala și centru optic (pentru 3D)
    f_px = estimate_focal_length_pixels(w, D435I_RGB_HFOV_DEG)
    cx, cy = w / 2.0, h / 2.0
    print(f"[INFO] Focală estimată: {f_px:.2f} px")

    # Video writers
    fourcc = cv2.VideoWriter_fourcc(*"XVID")
    det_writer = cv2.VideoWriter(OUTPUT_DET_VIDEO, fourcc, fps, (w, h), True)
    traj_writer = cv2.VideoWriter(OUTPUT_TRAJ_VIDEO, fourcc, fps, (w, h), True)
    topview_writer = cv2.VideoWriter(OUTPUT_TOPVIEW_VIDEO, fourcc, fps, (TOP_W, TOP_H), True)
    print("[OK] Video writers inițializați.")

    # CSV 3D + viteze
    csv_file = open(OUTPUT_CSV, mode="w", newline="")
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow([
        "frame_idx", "time_s",
        "u_px", "v_px", "r_px",
        "X_m", "Y_m", "Z_m", "distance_m",
        "Vx_m_s", "Vy_m_s", "Vz_m_s", "V_m_s"
    ])

    trajectory_points = []  # traiectorie în imagine
    prev_ball = None        # centru anterior pentru tracking 2D
    prev_3d = None          # poziție 3D anterioară pentru viteze
    prev_t = None           # timp anterior

    frame_idx = 0

    # reset video (am citit deja frame0)
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    while True:
        ok, frame = cap.read()
        if not ok or frame is None:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        t_s = frame_idx / fps  # timp absolut (secunde) pentru acest frame

        # --- 2) TRACKING CAMERĂ: template matching pe podea ---
        # WHAT: găsim unde se potrivește patch-ul de podea din frame0
        # WHY: deducem dx, dy (deplasarea camerei) față de frame0.
        res = cv2.matchTemplate(gray, floor_template, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, max_loc = cv2.minMaxLoc(res)
        match_x, match_y = max_loc

        dx = x_min - match_x
        dy = y_min - match_y

        # --- 3) Mască podea în cadrul curent (translatată cu -dx, -dy) ---
        # WHAT: reconstruim podeaua în frame curent.
        # WHY: vrem să știm unde e podeaua acum, pentru a filtra detecția mingii.
        M = np.float32([[1, 0, -dx],
                        [0, 1, -dy]])
        floor_mask_curr = cv2.warpAffine(
            floor_mask0, M, (w, h),
            flags=cv2.INTER_NEAREST,
            borderValue=0
        )

        det_frame = frame.copy()
        traj_frame = frame.copy()
        topview_frame = topview_accum.copy()

        # --- 4) DETECȚIE MINGE ROȘIE DOAR PE PODEA ---
        ball = detect_red_ball_on_floor(
            frame, floor_mask_curr,
            prev_center=prev_ball[:2] if prev_ball else None
        )

        X_m = Y_m = Z_m = dist_m = None
        Vx = Vy = Vz = V = None
        est = None

        if ball is not None:
            bx, by, br = ball
            prev_ball = ball

            # desenăm detecția în cadrul RGB
            cv2.circle(det_frame, (bx, by), br, (0, 255, 0), 2)
            cv2.circle(det_frame, (bx, by), 2, (0, 0, 255), -1)

            # --- 5) ESTIMARE 3D (X,Y,Z) + dist ---
            est = estimate_ball_3d(bx, by, br, f_px, cx, cy)
            if est is not None:
                X_m, Y_m, Z_m, dist_m = est
                txt = f"Dist={dist_m:.2f}m Z={Z_m:.2f}m"
                cv2.putText(det_frame, txt, (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                            (0, 255, 255), 2)

                # --- 6) VITEZE 3D (diferență de poziție / timp) ---
                if prev_3d is not None and prev_t is not None:
                    dT = t_s - prev_t
                    if dT > 0:
                        X_prev, Y_prev, Z_prev = prev_3d
                        Vx = (X_m - X_prev) / dT
                        Vy = (Y_m - Y_prev) / dT
                        Vz = (Z_m - Z_prev) / dT
                        V = math.sqrt(Vx * Vx + Vy * Vy + Vz * Vz)

                prev_3d = (X_m, Y_m, Z_m)
                prev_t = t_s

            # traiectorie în imagine (debug vizual)
            trajectory_points.append((bx, by))
            if len(trajectory_points) >= 2:
                for i in range(1, len(trajectory_points)):
                    cv2.line(traj_frame,
                             trajectory_points[i-1],
                             trajectory_points[i],
                             (0, 0, 255), 2)

            # --- 7) MAPARE MINGE ÎN COORDONATE FRAME0 ȘI APOI ÎN TOP-VIEW ---
            # WHAT:
            #   x_ref, y_ref = mingea în sistemul frame0.
            # WHY:
            #   vrem o coordonată "world" fixă, independentă de mișcarea camerei.
            x_ref = int(round(bx + dx))
            y_ref = int(round(by + dy))

            if 0 <= x_ref < w and 0 <= y_ref < h and floor_mask0[y_ref, x_ref] > 0:
                p0 = np.array([x_ref, y_ref, 1.0]).reshape(3, 1)
                p_top = H0_to_top @ p0
                if p_top[2, 0] != 0:
                    p_top /= p_top[2, 0]
                tx, ty = int(p_top[0, 0]), int(p_top[1, 0])
                if 0 <= tx < TOP_W and 0 <= ty < TOP_H:
                    # desenăm traiectoria mingii în world map 2D (top-view)
                    cv2.circle(topview_accum, (tx, ty), 3, (0, 0, 255), -1)
                    cv2.circle(topview_frame, (tx, ty), 3, (0, 0, 255), -1)

        # --- 8) POZIȚIA CAMEREI ÎN TOP-VIEW ---
        # WHAT:
        #   Considerăm centrul imaginii (cx,cy) + shift (dx,dy) ca "poziție a camerei" în frame0,
        #   apoi îl proiectăm în top-view.
        # WHY:
        #   vrem să vedem cum se mișcă camera relativ la podea și la mingea "fixată" în world map.
        cam_ref = np.array([cx + dx, cy + dy, 1.0]).reshape(3, 1)
        p_cam_top = H0_to_top @ cam_ref
        if p_cam_top[2, 0] != 0:
            p_cam_top /= p_cam_top[2, 0]
        cam_tx, cam_ty = int(p_cam_top[0, 0]), int(p_cam_top[1, 0])
        if 0 <= cam_tx < TOP_W and 0 <= cam_ty < TOP_H:
            cv2.circle(topview_frame, (cam_tx, cam_ty), 5, (255, 0, 0), 2)

        # --- 9) CSV: poziție + viteză 3D vs timp ---
        if ball is not None and est is not None:
            bx, by, br = ball

            def fmt(x):
                return "" if x is None else f"{x:.6f}"

            csv_writer.writerow([
                frame_idx, f"{t_s:.4f}",
                bx, by, br,
                fmt(X_m), fmt(Y_m), fmt(Z_m), fmt(dist_m),
                fmt(Vx), fmt(Vy), fmt(Vz), fmt(V)
            ])
        else:
            csv_writer.writerow([
                frame_idx, f"{t_s:.4f}",
                "", "", "",
                "", "", "", "",    # X,Y,Z,dist
                "", "", "", ""     # Vx,Vy,Vz,V
            ])

        # --- 10) scriem videouri ---
        det_writer.write(det_frame)
        traj_writer.write(traj_frame)
        topview_writer.write(topview_frame)

        frame_idx += 1

    cap.release()
    det_writer.release()
    traj_writer.release()
    topview_writer.release()
    csv_file.close()

    cv2.imwrite(OUTPUT_TOPVIEW_IMAGE, topview_accum)
    print(f"[OK] Top-view final salvat ca {OUTPUT_TOPVIEW_IMAGE}")
    print(f"[OK] Video detecție: {OUTPUT_DET_VIDEO}")
    print(f"[OK] Video traiectorie: {OUTPUT_TRAJ_VIDEO}")
    print(f"[OK] Video hartă 2D top-view: {OUTPUT_TOPVIEW_VIDEO}")
    print(f"[OK] CSV 3D + viteze: {OUTPUT_CSV}")
    print("[DONE] Pipeline complet DotLumen (2D+3D+velocities+top-view invariants).")


if __name__ == "__main__":
    main()
