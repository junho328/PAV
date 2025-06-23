"""
fewshot_retriever.py
--------------------
Task ↔ Macro Action Plan 예시를 끌어와서
few‑shot prompt를 만들어 주는 모듈
"""

from __future__ import annotations

import pathlib, pickle
from typing import List, Tuple, Dict

import faiss, numpy as np
from transformers import AutoModel

import argparse

from pathlib import Path
import json

class FewShotComposer:
    """
    FAISS‑기반 Task 유사도 검색 & 프롬프트 생성기
    """

    def __init__(self,
                 model_name: str = "jinaai/jina-embeddings-v3",
                 *,
                 app_name: str | None = None):
        self.model = AutoModel.from_pretrained(model_name,
                                               trust_remote_code=True).to("cuda")
        self.app_name = app_name              # ← 앱 이름별 분리 저장
        self.index: faiss.Index | None = None
        self.id2task: List[str] = []
        self.task2actions: Dict[str, str | list[str]] = {}

    # ─────────────────── 빌드 & 저장 ───────────────────
    def build_index(self,
                    task_action_pairs: List[Tuple[str, str | list[str]]]) -> None:
        self.id2task = [t for t, _ in task_action_pairs]
        self.task2actions = {t: a for t, a in task_action_pairs}

        emb = self._encode(self.id2task)              # (N, d) float32
        self.index = faiss.IndexFlatIP(emb.shape[1])  # cosine 유사도
        self.index.add(emb)

    def _store_dir(self, dir_: str | pathlib.Path) -> pathlib.Path:
        root = pathlib.Path(dir_)
        return root / self.app_name if self.app_name else root

    def save(self, dir_: str | pathlib.Path) -> None:
        assert self.index is not None, "index 가 없습니다. build_index 먼저!"
        store_path = self._store_dir(dir_)
        store_path.mkdir(parents=True, exist_ok=True)

        faiss.write_index(self.index, str(store_path / "shots.index"))
        with open(store_path / "meta.pkl", "wb") as f:
            pickle.dump({"id2task": self.id2task,
                         "task2actions": self.task2actions}, f)

    # ─────────────────── 로드 ───────────────────
    def load(self, dir_: str | pathlib.Path) -> None:
        store_path = self._store_dir(dir_)
        self.index = faiss.read_index(str(store_path / "shots.index"))
        with open(store_path / "meta.pkl", "rb") as f:
            meta = pickle.load(f)
        self.id2task = meta["id2task"]
        self.task2actions = meta["task2actions"]

    # ─────────────────── 검색 & 프롬프트 ───────────────────
    def query(self, task: str, k: int = 5) -> List[Tuple[str, str | list[str]]]:
        assert self.index is not None, "index 가 없습니다. build_index 또는 load!"
        emb = self._encode([task])
        _, I = self.index.search(emb, k + 5)  # 여유 있게 더 많이 검색해 둠

        seen = set()
        results = []
        for idx in I[0]:
            task_text = self.id2task[idx]
            if task_text not in seen and task_text != task:
                seen.add(task_text)
                results.append((task_text, self.task2actions[task_text]))
            if len(results) == k:
                break

        return results

    def build_prompt(self, task: str, *, k: int = 5) -> str:
        shots = self.query(task, k)
        segments: list[str] = []
        for idx, (t, actions) in enumerate(shots, start=1):
            segments.append(
                f"<Example {idx}>\nTask:\n{t}\nMacro Action Plan:\n{actions}"
            )
        return "\n\n".join(segments)

    # ─────────────────── 내부 헬퍼 ───────────────────
    def _encode(self, texts: List[str]) -> np.ndarray:
        emb = self.model.encode(texts,
                                task="text-matching",
                                convert_to_numpy=True)
        return emb.astype("float32")


# ─────────────────── 사용 예시 ───────────────────
if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--app_name", type=str, default="google_maps")
    parser.add_argument("--json_file", type=str, default="google_map_pool.json")
    args = parser.parse_args()
    
    # corpus = [
    #     ("Add sprite to the cart.", "[Search for sprite, Add to cart]"),
    #     ("Add Samyang ramen to the cart.", "[Search for Samyang ramen, Add to cart]"),
    #     ("Show the shipped item.", "[Open orders, Filter shipped, Select item]"),
    # ]
    
    corpus = []
    
    # text_path = Path(__file__).parent / "shot_pools" / args.app_name / args.text
    
    # with open(text_path, "r") as f:
    #     lines = f.readlines()
    #     for line in lines:
    #         line = line.strip()
    #         if line:
    #             task, action = line.split(" : ")
    #             corpus.append((task, action))
    
    json_path = Path(__file__).parent / "shot_pools" / args.app_name / args.json_file
    
    with open(json_path, "r") as f:
        data = json.load(f)
    
    for item in data:
        instruction = item["instruction"]
        macro_actions = item["macro_actions"]
        corpus.append((instruction, macro_actions))

    # (1) 인덱스 생성 & 저장
    composer = FewShotComposer(app_name=args.app_name)
    composer.build_index(corpus)
    composer.save("shot_pools")          # → faiss_store/google_maps/...

    # (2) 불러와서 프롬프트 생성
    loader = FewShotComposer(app_name=args.app_name)
    loader.load("shot_pools")

    # Example
    task = "Search for Namsan Tower."
    print(loader.build_prompt(task, k=3))
