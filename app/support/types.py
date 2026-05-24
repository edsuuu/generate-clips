from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class Word:
    text: str
    start: float
    end: float


@dataclass
class Segment:
    text: str
    start: float
    end: float
    words: list[Word] = field(default_factory=list)


@dataclass
class Transcript:
    language: str
    duration: float
    segments: list[Segment] = field(default_factory=list)

    @property
    def full_text(self) -> str:
        return " ".join(s.text.strip() for s in self.segments)


@dataclass
class Highlight:
    start: float
    end: float
    reason: str = ""
    score: float = 0.0

    @property
    def duration(self) -> float:
        return self.end - self.start


@dataclass
class CutMetadata:
    title: str
    description: str
    hashtags: list[str]


@dataclass
class Cut:
    index: int
    name: str  # PT1, PT2, ...
    highlight: Highlight
    video_path: Path
    metadata: CutMetadata | None = None


@dataclass
class VideoInfo:
    url: str
    video_id: str
    title: str
    duration: float
    file_path: Path


@dataclass
class CropPoint:
    """Centro do crop em coordenadas do vídeo original, em um instante."""
    timestamp: float
    x: int
    y: int


@dataclass
class CropTrajectory:
    """Trajetória do crop para um segmento de vídeo."""
    points: list[CropPoint] = field(default_factory=list)
    fallback_x: int = 0  # centro padrão se trajetória vazia
    fallback_y: int = 0

    def value_at(self, t: float) -> tuple[int, int]:
        """Retorna (x, y) interpolado linearmente no tempo t."""
        if not self.points:
            return self.fallback_x, self.fallback_y
        if t <= self.points[0].timestamp:
            return self.points[0].x, self.points[0].y
        if t >= self.points[-1].timestamp:
            return self.points[-1].x, self.points[-1].y
        for i in range(len(self.points) - 1):
            a, b = self.points[i], self.points[i + 1]
            if a.timestamp <= t <= b.timestamp:
                ratio = (t - a.timestamp) / (b.timestamp - a.timestamp) if b.timestamp > a.timestamp else 0
                x = int(a.x + (b.x - a.x) * ratio)
                y = int(a.y + (b.y - a.y) * ratio)
                return x, y
        return self.fallback_x, self.fallback_y
