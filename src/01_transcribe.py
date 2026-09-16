"""Local and resumable transcription of the Creceré AI call corpus."""

from __future__ import annotations

import argparse
import json
import os
import time
import wave
from pathlib import Path

from faster_whisper import WhisperModel
from huggingface_hub import snapshot_download

PROJECT = Path(__file__).resolve().parents[1]


def wav_duration(path: Path) -> float:
    with wave.open(str(path), "rb") as audio:
        return audio.getnframes() / audio.getframerate()


def load_done(output: Path) -> set[str]:
    if not output.exists():
        return set()
    done = set()
    for line in output.read_text(encoding="utf-8").splitlines():
        if line.strip():
            done.add(json.loads(line)["file"])
    return done


def ensure_model(repo_id: str, model_dir: Path) -> Path:
    """Use local_dir to avoid Windows symlink privileges in the HF cache."""
    if not (model_dir / "model.bin").exists():
        model_dir.mkdir(parents=True, exist_ok=True)
        snapshot_download(repo_id=repo_id, local_dir=str(model_dir), max_workers=2)
    return model_dir


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("--model-repo", default="Systran/faster-whisper-base")
    parser.add_argument("--model-dir", type=Path, default=PROJECT / ".cache" / "model_base")
    parser.add_argument("--output", type=Path, default=PROJECT / "data" / "interim" / "transcripts.jsonl")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--threads", type=int, default=max(1, (os.cpu_count() or 4) - 1))
    args = parser.parse_args()

    expected = {
        "audios_humanos_censurados": "Humano",
        "audios_ia_censurados": "IA",
    }
    files = []
    for folder, group in expected.items():
        source = args.data_root / folder
        if not source.exists():
            raise SystemExit(f"No existe la carpeta esperada: {source}")
        for path in sorted(source.glob("*.wav")):
            files.append((group, path, wav_duration(path)))

    files.sort(key=lambda row: row[2])
    done = load_done(args.output)
    pending = [row for row in files if row[1].name not in done]
    if args.limit:
        pending = pending[: args.limit]

    model_dir = ensure_model(args.model_repo, args.model_dir)
    model = WhisperModel(
        str(model_dir),
        device="cpu",
        compute_type="int8",
        cpu_threads=args.threads,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)

    started = time.time()
    processed_audio = 0.0
    for index, (group, path, duration) in enumerate(pending, start=1):
        t0 = time.time()
        segments_iter, info = model.transcribe(
            str(path),
            language="es",
            beam_size=1,
            best_of=1,
            vad_filter=True,
            vad_parameters={"min_silence_duration_ms": 500},
            condition_on_previous_text=False,
            temperature=0,
        )
        segments = [
            {
                "start": round(segment.start, 2),
                "end": round(segment.end, 2),
                "text": segment.text.strip(),
                "avg_logprob": round(segment.avg_logprob, 4),
                "no_speech_prob": round(segment.no_speech_prob, 4),
            }
            for segment in segments_iter
        ]
        row = {
            "file": path.name,
            "group": group,
            "duration_s": round(duration, 2),
            "language": info.language,
            "language_probability": round(info.language_probability, 4),
            "segments": segments,
            "text": " ".join(item["text"] for item in segments),
        }
        with args.output.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")

        elapsed = time.time() - t0
        processed_audio += duration
        speed = processed_audio / max(time.time() - started, 1e-6)
        eta_s = sum(item[2] for item in pending[index:]) / max(speed, 1e-6)
        print(
            f"[{index}/{len(pending)}] {group} {path.name} "
            f"audio={duration:.0f}s wall={elapsed:.1f}s x{speed:.2f} eta={eta_s / 60:.1f}m",
            flush=True,
        )


if __name__ == "__main__":
    main()
