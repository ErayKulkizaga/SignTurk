"""
landmark_smoother.py — Landmark JSON Smoothing Pipeline
=======================================================
3 asamali smoothing:
  1) Null frame interpolation — kayip el verilerini komsu frame'lerden doldurur
  2) Outlier spike removal   — ani Z ziplamalari tespit edip duzeltir
  3) Gaussian temporal smooth — tum eksenlerde (ozellikle Z) gurultuyu temizler

Backend'de /landmark/{word} endpoint'inden veri donmeden once uygulanir.
"""

import json, copy, math
from typing import List, Dict, Any, Optional

# ── Konfigurasyon ──────────────────────────────────────────────
GAUSSIAN_SIGMA   = 1.2    # Temporal smoothing kuvveti (frame cinsinden)
GAUSSIAN_WINDOW  = 5      # Kac frame saga sola bakacak (2*W+1 pencere)
SPIKE_THRESHOLD  = 0.15   # Frame arasi bu kadar ziplayan Z degeri spike sayilir
SPIKE_Z_WEIGHT   = 1.0    # Z eksenine ekstra hassasiyet (X,Y icin 1.0)
MIN_INTERP_GAP   = 1      # Minimum bosluk buyuklugu (1 = tek frame bosluk bile doldur)
MAX_INTERP_GAP   = 20     # Uzun bosluklari da doldur (pose wrist verisinden destek alarak)


def smooth_landmark_data(raw_data: dict) -> dict:
    """
    Ana pipeline: raw landmark JSON -> smoothed landmark JSON
    Orijinal veriyi degistirmez, yeni kopya doner.
    """
    data = copy.deepcopy(raw_data)
    frames = data.get("frames", [])
    if len(frames) < 3:
        return data

    # Asama 1: Null frame'leri interpole et
    _interpolate_null_hands(frames, "left_hand")
    _interpolate_null_hands(frames, "right_hand")

    # Asama 2: Spike removal (pose ve eller icin)
    _remove_spikes_pose(frames)
    _remove_spikes_hand(frames, "left_hand")
    _remove_spikes_hand(frames, "right_hand")

    # Asama 3: Gaussian temporal smoothing
    _gaussian_smooth_pose(frames)
    _gaussian_smooth_hand(frames, "left_hand")
    _gaussian_smooth_hand(frames, "right_hand")

    data["frames"] = frames
    data["smoothed"] = True
    return data


# ══════════════════════════════════════════════════════════════
# Asama 1: Null Frame Interpolation
# ══════════════════════════════════════════════════════════════

def _interpolate_null_hands(frames: List[dict], hand_key: str):
    """
    Kayip el frame'lerini komsu gecerli frame'lerden lineer interpolasyonla doldurur.
    Ornek: [OK, NULL, NULL, OK] -> ortadaki 2 frame lineer gecis olur.
    Eger bosluk cok uzunsa ve hic gecerli frame yoksa, pose wrist verisinden
    en yakin gecerli el'i shift ederek sentetik el olusturur.
    """
    n = len(frames)
    # Gecerli (non-null) frame indekslerini bul
    valid_indices = [i for i in range(n) if frames[i].get(hand_key)]

    if len(valid_indices) == 0:
        return  # Bu elde hic veri yok, yapacak bir sey yok

    if len(valid_indices) == 1:
        # Tek gecerli frame var — hepsini onunla doldur
        ref = frames[valid_indices[0]][hand_key]
        for i in range(n):
            if not frames[i].get(hand_key):
                frames[i][hand_key] = copy.deepcopy(ref)
        return

    # Her boslugu bul ve doldur
    for vi in range(len(valid_indices) - 1):
        start_idx = valid_indices[vi]
        end_idx   = valid_indices[vi + 1]
        gap = end_idx - start_idx - 1

        if gap < MIN_INTERP_GAP or gap > MAX_INTERP_GAP:
            continue

        start_hand = frames[start_idx][hand_key]
        end_hand   = frames[end_idx][hand_key]

        # Uzun bosluklarda: pose wrist hareketini de hesaba kat
        wrist_key = "15" if hand_key == "left_hand" else "16"
        start_wrist = frames[start_idx].get("pose", {}).get(wrist_key)
        end_wrist   = frames[end_idx].get("pose", {}).get(wrist_key)

        for g in range(1, gap + 1):
            t = g / (gap + 1)
            fill_idx = start_idx + g

            interp_hand = _lerp_hand(start_hand, end_hand, t)

            # Uzun boslukta pose wrist pozisyonundan ek duzeltme
            if gap > 6 and start_wrist and end_wrist:
                mid_wrist = frames[fill_idx].get("pose", {}).get(wrist_key)
                if mid_wrist:
                    # Beklenen wrist pozisyonu (lineer interp)
                    expected_wx = start_wrist[0] + (end_wrist[0] - start_wrist[0]) * t
                    expected_wy = start_wrist[1] + (end_wrist[1] - start_wrist[1]) * t
                    # Gercek wrist ile beklenen arasindaki fark
                    dx = mid_wrist[0] - expected_wx
                    dy = mid_wrist[1] - expected_wy
                    # Bu farki el landmark'larina ekle (shift)
                    for lm in interp_hand:
                        lm[0] += dx
                        lm[1] += dy

            frames[fill_idx][hand_key] = interp_hand

    # Baslangictaki null'lari ilk gecerli frame ile doldur (hold)
    first_valid = valid_indices[0]
    for i in range(first_valid):
        if not frames[i].get(hand_key):
            frames[i][hand_key] = copy.deepcopy(frames[first_valid][hand_key])

    # Sondaki null'lari son gecerli frame ile doldur (hold)
    last_valid = valid_indices[-1]
    for i in range(last_valid + 1, n):
        if not frames[i].get(hand_key):
            frames[i][hand_key] = copy.deepcopy(frames[last_valid][hand_key])


def _lerp_hand(hand_a: List, hand_b: List, t: float) -> List:
    """21 landmark noktasini lineer interpolasyonla karistir."""
    result = []
    for i in range(min(len(hand_a), len(hand_b))):
        pa, pb = hand_a[i], hand_b[i]
        result.append([
            pa[0] + (pb[0] - pa[0]) * t,
            pa[1] + (pb[1] - pa[1]) * t,
            pa[2] + (pb[2] - pa[2]) * t,
        ])
    return result


# ══════════════════════════════════════════════════════════════
# Asama 2: Spike Removal
# ══════════════════════════════════════════════════════════════

def _remove_spikes_pose(frames: List[dict]):
    """Pose landmark'larindaki ani ziplama spike'larini tespit edip duzeltir."""
    n = len(frames)
    if n < 3:
        return

    # Her pose landmark key'i icin
    all_keys = set()
    for fr in frames:
        all_keys.update(fr.get("pose", {}).keys())

    for key in all_keys:
        # Bu key'in tum frame'lerdeki degerlerini topla
        values = []
        for fr in frames:
            v = fr.get("pose", {}).get(key)
            values.append(v if v else None)

        # Spike tespit: ortadaki frame sagdan ve soldan cok farkliysa spike
        for i in range(1, n - 1):
            if values[i] is None or values[i-1] is None or values[i+1] is None:
                continue

            for axis in range(3):
                prev_v = values[i-1][axis]
                curr_v = values[i][axis]
                next_v = values[i+1][axis]

                # Threshold: Z icin daha hassas
                thresh = SPIKE_THRESHOLD * (SPIKE_Z_WEIGHT if axis == 2 else 1.0)

                diff_prev = abs(curr_v - prev_v)
                diff_next = abs(curr_v - next_v)
                diff_pn   = abs(next_v - prev_v)

                # Spike: hem oncekinden hem sonrakinden uzak, ama onceki-sonraki birbirine yakin
                if diff_prev > thresh and diff_next > thresh and diff_pn < thresh:
                    # Spike! Ortalamayla degistir
                    values[i][axis] = (prev_v + next_v) / 2.0
                    frames[i]["pose"][key][axis] = values[i][axis]


def _remove_spikes_hand(frames: List[dict], hand_key: str):
    """El landmark'larindaki spike'lari temizle."""
    n = len(frames)
    if n < 3:
        return

    for lm_idx in range(21):
        for i in range(1, n - 1):
            hand_prev = frames[i-1].get(hand_key)
            hand_curr = frames[i].get(hand_key)
            hand_next = frames[i+1].get(hand_key)
            if not hand_prev or not hand_curr or not hand_next:
                continue

            for axis in range(3):
                prev_v = hand_prev[lm_idx][axis]
                curr_v = hand_curr[lm_idx][axis]
                next_v = hand_next[lm_idx][axis]

                thresh = SPIKE_THRESHOLD * (SPIKE_Z_WEIGHT if axis == 2 else 1.0)
                diff_prev = abs(curr_v - prev_v)
                diff_next = abs(curr_v - next_v)
                diff_pn   = abs(next_v - prev_v)

                if diff_prev > thresh and diff_next > thresh and diff_pn < thresh:
                    frames[i][hand_key][lm_idx][axis] = (prev_v + next_v) / 2.0


# ══════════════════════════════════════════════════════════════
# Asama 3: Gaussian Temporal Smoothing
# ══════════════════════════════════════════════════════════════

def _make_gaussian_kernel(sigma: float, window: int) -> List[float]:
    """1D Gaussian kernel olustur, toplami 1'e normalize et."""
    kernel = []
    for i in range(-window, window + 1):
        kernel.append(math.exp(-0.5 * (i / sigma) ** 2))
    total = sum(kernel)
    return [k / total for k in kernel]


_KERNEL = _make_gaussian_kernel(GAUSSIAN_SIGMA, GAUSSIAN_WINDOW)


def _gaussian_smooth_series(values: List[Optional[List[float]]]) -> List[Optional[List[float]]]:
    """
    [x,y,z] degerlerinden olusan bir seri uzerinde Gaussian smoothing uygula.
    None degerler atlanir.
    """
    n = len(values)
    if n < 3:
        return values

    result = [None] * n
    w = GAUSSIAN_WINDOW

    for i in range(n):
        if values[i] is None:
            result[i] = None
            continue

        dim = len(values[i])
        smoothed = [0.0] * dim
        weight_sum = 0.0

        for ki, k in enumerate(range(-w, w + 1)):
            j = i + k
            if j < 0 or j >= n or values[j] is None:
                continue
            weight = _KERNEL[ki]
            weight_sum += weight
            for d in range(dim):
                smoothed[d] += values[j][d] * weight

        if weight_sum > 0:
            result[i] = [s / weight_sum for s in smoothed]
        else:
            result[i] = values[i][:]

    return result


def _gaussian_smooth_pose(frames: List[dict]):
    """Pose landmark'larina Gaussian smoothing uygula."""
    n = len(frames)
    all_keys = set()
    for fr in frames:
        all_keys.update(fr.get("pose", {}).keys())

    for key in all_keys:
        series = []
        for fr in frames:
            v = fr.get("pose", {}).get(key)
            series.append(v[:] if v else None)  # Kopya al

        smoothed = _gaussian_smooth_series(series)

        for i in range(n):
            if smoothed[i] is not None and key in frames[i].get("pose", {}):
                frames[i]["pose"][key] = smoothed[i]


def _gaussian_smooth_hand(frames: List[dict], hand_key: str):
    """El landmark'larina Gaussian smoothing uygula."""
    n = len(frames)

    for lm_idx in range(21):
        series = []
        for fr in frames:
            hand = fr.get(hand_key)
            if hand and len(hand) > lm_idx:
                series.append(hand[lm_idx][:])  # Kopya
            else:
                series.append(None)

        smoothed = _gaussian_smooth_series(series)

        for i in range(n):
            hand = frames[i].get(hand_key)
            if smoothed[i] is not None and hand and len(hand) > lm_idx:
                frames[i][hand_key][lm_idx] = smoothed[i]


# ══════════════════════════════════════════════════════════════
# Test / Debug
# ══════════════════════════════════════════════════════════════

def compare_stats(raw_frames, smooth_frames, label=""):
    """Smoothing oncesi-sonrasi karsilastirma istatistikleri."""
    def z_jitter(frs, pose_key="15"):
        diffs = []
        for i in range(1, len(frs)):
            p = frs[i].get("pose", {}).get(pose_key)
            pp = frs[i-1].get("pose", {}).get(pose_key)
            if p and pp:
                diffs.append(abs(p[2] - pp[2]))
        return diffs

    raw_j = z_jitter(raw_frames)
    smooth_j = z_jitter(smooth_frames)

    if raw_j and smooth_j:
        import numpy as np
        print(f"[{label}] Pose Z jitter — Raw: mean={np.mean(raw_j):.5f} max={np.max(raw_j):.5f} | "
              f"Smooth: mean={np.mean(smooth_j):.5f} max={np.max(smooth_j):.5f}")

    # Null el sayisi
    for hk in ["left_hand", "right_hand"]:
        raw_null = sum(1 for f in raw_frames if not f.get(hk))
        smo_null = sum(1 for f in smooth_frames if not f.get(hk))
        if raw_null > 0:
            print(f"[{label}] {hk} null frames — Raw: {raw_null} | Smooth: {smo_null}")


if __name__ == "__main__":
    import sys, os, numpy as np

    # Test: uploads klasorundan tum JSON'lari isle
    test_dir = sys.argv[1] if len(sys.argv) > 1 else "."
    files = sorted(f for f in os.listdir(test_dir) if f.endswith(".json"))

    for fname in files:
        path = os.path.join(test_dir, fname)
        try:
            with open(path) as f:
                raw = json.load(f)
            if "frames" not in raw:
                continue

            smoothed = smooth_landmark_data(raw)
            compare_stats(raw["frames"], smoothed["frames"], fname)
            print()
        except Exception as e:
            print(f"[{fname}] HATA: {e}")
