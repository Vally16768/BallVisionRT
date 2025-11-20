import cv2
import numpy as np
from sklearn.cluster import KMeans


# ==========================
# CONFIG
# ==========================
VIDEO_PATH = "rgb.avi"          # pune aici calea spre filmarea ta
OUTPUT_FIRST_FRAME = "frame0.png"
OUTPUT_FLOOR_MASK = "floor_mask.png"
OUTPUT_FLOOR_OVERLAY = "floor_overlay.png"
OUTPUT_FLOOR_TOPVIEW = "floor_topview.png"  # se folosește doar dacă umpli manual punctele


# ==========================
# 1. Citește primul frame din video
# ==========================
def read_first_frame(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Nu pot deschide video: {video_path}")

    ok, frame = cap.read()
    cap.release()

    if not ok or frame is None:
        raise RuntimeError("Nu am reușit să citesc primul frame din video.")

    return frame


# ==========================
# 2. Segmentează automat podeaua folosind K-Means
#    Ideea: imaginea e împărțită în clustere de culoare;
#    alegem clusterul care:
#       - are cea mai mare parte din pixeli în jumătatea de jos
#       - este suficient de extins
#       - nu este foarte luminos (ca pereții albi / cerul)
# ==========================
def segment_floor_kmeans(bgr_img, k=4):
    h, w = bgr_img.shape[:2]

    # micșorăm imaginea pentru K-Means (accelerează)
    scale = 0.25  # 25% din rezoluție
    small = cv2.resize(bgr_img, (int(w * scale), int(h * scale)))
    sh, sw = small.shape[:2]

    # date pentru KMeans: Nx3 (BGR)
    data = small.reshape(-1, 3).astype(np.float32)

    # rulăm K-Means
    kmeans = KMeans(n_clusters=k, n_init=5, random_state=0)
    kmeans.fit(data)
    labels = kmeans.labels_.reshape(sh, sw)

    # convertim în LAB pentru a avea luminozitatea (L)
    lab_small = cv2.cvtColor(small, cv2.COLOR_BGR2LAB)
    L = lab_small[:, :, 0]

    # alegem clusterul "cel mai probabil podea"
    best_score = -1.0
    best_idx = 0

    for i in range(k):
        ys, xs = np.where(labels == i)
        if ys.size == 0:
            continue

        frac_bottom = ys.mean() / sh           # 0 sus, 1 jos
        area_frac = ys.size / float(sh * sw)   # procent din imagine
        mean_L = L[ys, xs].mean() / 255.0      # 0..1, 1 = foarte luminos

        # scor euristic: preferim cluster
        #   - mai jos în imagine
        #   - mare ca arie
        #   - nu foarte luminos (penalizăm luminozitatea)
        score = frac_bottom + 0.5 * area_frac - 0.3 * mean_L

        # print(f"cluster {i}: score={score:.3f}")

        if score > best_score:
            best_score = score
            best_idx = i

    # mască în versiunea mică
    floor_mask_small = (labels == best_idx).astype(np.uint8) * 255

    # extrapolăm masca la rezoluția originală
    floor_mask = cv2.resize(
        floor_mask_small, (w, h), interpolation=cv2.INTER_NEAREST
    )

    # operații morfologice pentru a curăța masca
    kernel = np.ones((25, 25), np.uint8)
    floor_mask = cv2.morphologyEx(floor_mask, cv2.MORPH_CLOSE, kernel)
    floor_mask = cv2.morphologyEx(floor_mask, cv2.MORPH_OPEN, kernel)

    # păstrăm doar cel mai mare obiect (componenta cea mai mare)
    contours, _ = cv2.findContours(
        floor_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    if not contours:
        raise RuntimeError("Nu am putut găsi niciun contur pentru podea.")

    largest = max(contours, key=cv2.contourArea)
    mask_clean = np.zeros_like(floor_mask)
    cv2.drawContours(mask_clean, [largest], -1, 255, thickness=-1)

    return mask_clean


# ==========================
# 3. Creează overlay pentru debug (podea colorată în verde)
# ==========================
def create_overlay(frame, floor_mask):
    overlay = frame.copy()
    green = np.zeros_like(frame)
    green[:, :, 1] = 255  # canal G

    # combinăm: 30% verde + 70% imagine originală pe zonele de podea
    alpha = 0.3
    mask_3ch = cv2.merge([floor_mask, floor_mask, floor_mask]) / 255.0
    overlay = (frame * (1 - alpha * mask_3ch) + green * (alpha * mask_3ch)).astype(
        np.uint8
    )
    return overlay


# ==========================
# 4. (OPȚIONAL) Proiecție top-view dintr-o perspectivă simplă
#
#    AICI trebuie să setezi manual 4 puncte în imagine (colțurile podelei)
#    dacă vrei o hartă "bird-eye". Automat, fără niciun fel de informații
#    geometrice, nu ai cum să le deduci perfect dintr-o singură imagine.
#
#    Dacă nu te interesează bird-eye, poți ignora funcția asta: harta 2D
#    este deja "floor_mask.png".
# ==========================
def compute_topview(frame, floor_mask):
    h, w = frame.shape[:2]

    # ---- MOD MANUAL: pune coordonatele celor 4 colțuri ale podelei ----
    # Ordinea: [stânga-sus, dreapta-sus, dreapta-jos, stânga-jos]
    # Exemplu fictiv – TREBUIE să le măsori din imaginea ta!
    src = np.float32([
        [100, 300],   # stânga-sus
        [w - 100, 300],  # dreapta-sus
        [w - 50, h - 50],  # dreapta-jos
        [50, h - 50],   # stânga-jos
    ])

    # dimensiunea hărții 2D top-view (în pixeli)
    top_w = 800
    top_h = 600
    dst = np.float32([
        [0, 0],
        [top_w - 1, 0],
        [top_w - 1, top_h - 1],
        [0, top_h - 1],
    ])

    # matrice de homografie
    H = cv2.getPerspectiveTransform(src, dst)

    # proiecție bird-eye pentru mască
    topview_mask = cv2.warpPerspective(
        floor_mask, H, (top_w, top_h), flags=cv2.INTER_NEAREST
    )

    return topview_mask


# ==========================
# MAIN
# ==========================
def main():
    # 1) primul frame
    frame = read_first_frame(VIDEO_PATH)
    cv2.imwrite(OUTPUT_FIRST_FRAME, frame)
    print(f"[OK] Primul frame salvat ca {OUTPUT_FIRST_FRAME}")

    # 2) segmentarea automată a podelei
    floor_mask = segment_floor_kmeans(frame)
    cv2.imwrite(OUTPUT_FLOOR_MASK, floor_mask)
    print(f"[OK] Mască podea salvată ca {OUTPUT_FLOOR_MASK}")

    # 3) overlay pentru verificare vizuală
    overlay = create_overlay(frame, floor_mask)
    cv2.imwrite(OUTPUT_FLOOR_OVERLAY, overlay)
    print(f"[OK] Overlay salvat ca {OUTPUT_FLOOR_OVERLAY}")

    # 4) (opțional) hartă 2D top-view folosind homografie
    #    – dacă NU vrei bird-eye sau nu pui manual colțurile,
    #      poți comenta codul de mai jos.
    try:
        topview = compute_topview(frame, floor_mask)
        cv2.imwrite(OUTPUT_FLOOR_TOPVIEW, topview)
        print(f"[OK] Hartă 2D top-view salvată ca {OUTPUT_FLOOR_TOPVIEW}")
    except Exception as e:
        print(
            "[WARN] Nu am putut genera top-view automat. "
            "Verifică punctele src în compute_topview()."
        )
        print("Detalii:", e)


if __name__ == "__main__":
    main()
