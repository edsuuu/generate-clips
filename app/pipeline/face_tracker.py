"""Face tracking com Active Speaker Detection (MediaPipe Tasks API).

Estratégia:
1. Sample frames do vídeo a N fps.
2. MediaPipe Face Detector localiza bboxes.
3. MediaPipe Face Landmarker pega landmarks dos lábios para cada face.
4. Para cada frame, calcula 'lip activity' (variação da abertura da boca entre
   frames consecutivos) por face.
5. Calcula energia do áudio em janela ao redor do frame (RMS via librosa).
6. Para cada instante, escolhe como 'speaker' a face com maior produto
   lip_activity * audio_energy + bias de tamanho. Se nenhuma face detectada,
   mantém a última posição conhecida ou o centro como fallback.
7. Suaviza a trajetória do centro do crop (savgol filter).
"""

from __future__ import annotations

import os
import subprocess
from dataclasses import dataclass
from pathlib import Path

# Reduz ruído nativo de MediaPipe/TFLite no stderr do processo.
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("GLOG_minloglevel", "3")

import cv2  # type: ignore
import mediapipe as mp  # type: ignore
import numpy as np
from mediapipe.tasks import python as mp_tasks  # type: ignore
from mediapipe.tasks.python import vision as mp_vision  # type: ignore

from app.support.config import settings
from app.support.logger import logger
from app.support.types import CropPoint, CropTrajectory

# app/pipeline/face_tracker.py → raiz é parent.parent.parent
MODELS_DIR = Path(__file__).resolve().parent.parent.parent / "models"
FACE_DETECTOR_MODEL = MODELS_DIR / "blaze_face_short_range.tflite"
FACE_LANDMARKER_MODEL = MODELS_DIR / "face_landmarker.task"

# Índices de landmarks de Face Mesh para os lábios (modelo 478 pontos).
# Subset chave: contornos central superior e inferior da boca.
LIPS_TOP_IDX = [13, 312, 311, 310, 415]
LIPS_BOTTOM_IDX = [14, 317, 402, 318, 324]


@dataclass
class _FaceObservation:
    timestamp: float
    cx: int
    cy: int
    bbox_w: int
    bbox_h: int
    lip_open: float


class FaceTracker:
    def __init__(self, sample_fps: int | None = None):
        self.sample_fps = sample_fps or settings.face_tracking_sample_fps

        if not FACE_DETECTOR_MODEL.exists() or not FACE_LANDMARKER_MODEL.exists():
            raise FileNotFoundError(
                f"Modelos MediaPipe não encontrados em {MODELS_DIR}. "
                "Baixe blaze_face_short_range.tflite e face_landmarker.task."
            )

        fd_options = mp_vision.FaceDetectorOptions(
            base_options=mp_tasks.BaseOptions(model_asset_path=str(FACE_DETECTOR_MODEL)),
            running_mode=mp_vision.RunningMode.IMAGE,
            min_detection_confidence=0.5,
        )
        self._face_detector = mp_vision.FaceDetector.create_from_options(fd_options)

        fl_options = mp_vision.FaceLandmarkerOptions(
            base_options=mp_tasks.BaseOptions(model_asset_path=str(FACE_LANDMARKER_MODEL)),
            running_mode=mp_vision.RunningMode.IMAGE,
            num_faces=4,
            min_face_detection_confidence=0.5,
            min_face_presence_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self._face_landmarker = mp_vision.FaceLandmarker.create_from_options(fl_options)

    def track_segment(
        self,
        video_path: Path,
        start: float,
        end: float,
    ) -> CropTrajectory:
        """Calcula trajetória de crop para o segmento [start, end] do vídeo."""
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise RuntimeError(f"Não foi possível abrir vídeo: {video_path}")

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        trajectory = CropTrajectory(fallback_x=width // 2, fallback_y=height // 2)

        sample_period = 1.0 / max(1, self.sample_fps)
        timestamps = np.arange(start, end, sample_period)

        audio_energy = self._extract_audio_energy(video_path, start, end, timestamps)

        prev_lip_opens: dict[int, float] = {}
        observations_per_t: list[list[_FaceObservation]] = []

        for ts in timestamps:
            cap.set(cv2.CAP_PROP_POS_MSEC, ts * 1000.0)
            ok, frame = cap.read()
            if not ok:
                observations_per_t.append([])
                continue

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

            faces = self._detect_faces(mp_image, width, height)
            if not faces:
                observations_per_t.append([])
                continue

            landmark_result = self._face_landmarker.detect(mp_image)
            face_landmarks_list = landmark_result.face_landmarks or []

            frame_obs: list[_FaceObservation] = []
            for face_idx, (cx, cy, w, h) in enumerate(faces):
                lip_open = 0.0
                if face_idx < len(face_landmarks_list):
                    landmarks = face_landmarks_list[face_idx]
                    lip_open = self._lip_openness(landmarks)
                frame_obs.append(
                    _FaceObservation(
                        timestamp=float(ts),
                        cx=cx,
                        cy=cy,
                        bbox_w=w,
                        bbox_h=h,
                        lip_open=lip_open,
                    )
                )
            observations_per_t.append(frame_obs)

        cap.release()

        last_known: tuple[int, int] | None = None
        for i, (ts, frame_obs) in enumerate(zip(timestamps, observations_per_t, strict=False)):
            audio_e = audio_energy[i] if i < len(audio_energy) else 0.0
            chosen, last_known = self._pick_speaker(frame_obs, audio_e, prev_lip_opens, last_known)
            if chosen is None:
                if last_known is not None:
                    trajectory.points.append(
                        CropPoint(timestamp=float(ts), x=last_known[0], y=last_known[1])
                    )
                continue
            trajectory.points.append(CropPoint(timestamp=float(ts), x=chosen.cx, y=chosen.cy))

        trajectory = self._smooth(trajectory, width, height)

        logger.info(
            f"Face tracking [{start:.1f}s-{end:.1f}s]: {len(trajectory.points)} pontos amostrados"
        )
        return trajectory

    def _detect_faces(self, mp_image, width: int, height: int) -> list[tuple[int, int, int, int]]:
        result = self._face_detector.detect(mp_image)
        if not result.detections:
            return []
        faces = []
        for det in result.detections:
            bbox = det.bounding_box
            x = max(0, int(bbox.origin_x))
            y = max(0, int(bbox.origin_y))
            w = int(bbox.width)
            h = int(bbox.height)
            cx = x + w // 2
            cy = y + h // 2
            faces.append((cx, cy, w, h))
        faces.sort(key=lambda f: f[2] * f[3], reverse=True)
        return faces

    def _pick_speaker(
        self,
        frame_obs: list[_FaceObservation],
        audio_e: float,
        prev_lip_opens: dict[int, float],
        last_known: tuple[int, int] | None,
    ) -> tuple[_FaceObservation | None, tuple[int, int] | None]:
        if not frame_obs:
            return None, last_known

        chosen = frame_obs[0]
        if len(frame_obs) > 1:
            chosen = self._choose_best_face(frame_obs, audio_e, prev_lip_opens)

        for face_idx, obs in enumerate(frame_obs):
            prev_lip_opens[face_idx] = obs.lip_open

        return chosen, (chosen.cx, chosen.cy)

    def _choose_best_face(
        self,
        frame_obs: list[_FaceObservation],
        audio_e: float,
        prev_lip_opens: dict[int, float],
    ) -> _FaceObservation:
        best_score = -1.0
        chosen = frame_obs[0]
        for face_idx, obs in enumerate(frame_obs):
            prev = prev_lip_opens.get(face_idx, obs.lip_open)
            lip_activity = abs(obs.lip_open - prev)
            size_score = obs.bbox_w * obs.bbox_h
            speaker_score = lip_activity * audio_e * 1e6
            score = size_score * 0.3 + speaker_score
            if score > best_score:
                best_score = score
                chosen = obs
        return chosen

    def _lip_openness(self, landmarks) -> float:
        top_y = sum(landmarks[i].y for i in LIPS_TOP_IDX) / len(LIPS_TOP_IDX)
        bot_y = sum(landmarks[i].y for i in LIPS_BOTTOM_IDX) / len(LIPS_BOTTOM_IDX)
        return float(abs(bot_y - top_y))

    def _extract_audio_energy(
        self,
        video_path: Path,
        start: float,
        end: float,
        timestamps: np.ndarray,
    ) -> np.ndarray:
        try:
            sr = 16000
            duration = max(0.0, end - start)
            if duration <= 0:
                return np.zeros(len(timestamps))

            cmd = [
                "ffmpeg",
                "-nostdin",
                "-v",
                "error",
                "-ss",
                f"{start:.3f}",
                "-t",
                f"{duration:.3f}",
                "-i",
                str(video_path),
                "-vn",
                "-ac",
                "1",
                "-ar",
                str(sr),
                "-f",
                "s16le",
                "-acodec",
                "pcm_s16le",
                "pipe:1",
            ]
            result = subprocess.run(cmd, capture_output=True, check=False)
            if result.returncode != 0:
                stderr = result.stderr.decode("utf-8", errors="ignore").strip()
                raise RuntimeError(stderr or "ffmpeg retornou erro ao extrair áudio")

            y = np.frombuffer(result.stdout, dtype=np.int16).astype(np.float32)
            if y.size == 0:
                return np.zeros(len(timestamps))
            y /= 32768.0
        except Exception as e:
            logger.warning(f"Falha ao carregar áudio para ASD: {e}")
            return np.zeros(len(timestamps))

        window = int(0.1 * sr)
        energies = []
        for ts in timestamps:
            center = int((ts - start) * sr)
            lo = max(0, center - window // 2)
            hi = min(len(y), center + window // 2)
            chunk = y[lo:hi]
            if len(chunk) == 0:
                energies.append(0.0)
            else:
                energies.append(float(np.sqrt(np.mean(chunk**2))))

        e = np.array(energies)
        if e.max() > 0:
            e = e / e.max()
        return e

    def _smooth(self, trajectory: CropTrajectory, width: int, height: int) -> CropTrajectory:
        if len(trajectory.points) < 5:
            return trajectory

        try:
            from scipy.signal import savgol_filter  # type: ignore
        except ImportError:
            return trajectory

        xs = np.array([p.x for p in trajectory.points], dtype=float)
        ys = np.array([p.y for p in trajectory.points], dtype=float)

        window = min(len(xs), 11)
        if window % 2 == 0:
            window -= 1
        poly = min(3, window - 1)

        xs_s = savgol_filter(xs, window, poly)
        ys_s = savgol_filter(ys, window, poly)

        smoothed = CropTrajectory(
            fallback_x=trajectory.fallback_x,
            fallback_y=trajectory.fallback_y,
        )
        for p, xs_v, ys_v in zip(trajectory.points, xs_s, ys_s, strict=False):
            smoothed.points.append(
                CropPoint(
                    timestamp=p.timestamp,
                    x=int(np.clip(xs_v, 0, width)),
                    y=int(np.clip(ys_v, 0, height)),
                )
            )
        return smoothed
