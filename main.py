"""Photography Agent pipeline entrypoint."""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any

from src import config
from src.clustering import ImageClusterer
from src.llm_judge import LLMJudge
from src.perception import ImagePerceiver
from src.scorer import AestheticScorer


def _find_photo_path(photo_id: str, raw_dir: Path = config.RAW_PHOTOS_DIR) -> Path | None:
    for suffix in config.IMAGE_EXTENSIONS:
        candidate = raw_dir / f"{photo_id}{suffix}"
        if candidate.exists():
            return candidate
        upper_candidate = raw_dir / f"{photo_id}{suffix.upper()}"
        if upper_candidate.exists():
            return upper_candidate
    matches = sorted(raw_dir.glob(f"{photo_id}.*"))
    return matches[0] if matches else None


def _write_results(payload: dict[str, Any]) -> Path:
    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = config.RESULTS_DIR / f"selection_{stamp}.json"
    output_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return output_path


def run_pipeline(skip_llm: bool = False, top_n: int = config.SCORER_TOP_N) -> dict[str, Any]:
    print("\n🚀 本地智能摄影选片 Agent 启动")
    print(f"📁 原图目录: {config.RAW_PHOTOS_DIR}")
    print(f"🧠 特征缓存: {config.CACHE_DIR}\n")

    print("========== [阶段一] 感知层 / CLIP 特征提取 ==========")
    perceiver = ImagePerceiver()
    perception_stats = perceiver.extract_and_save_features(
        input_dir=config.RAW_PHOTOS_DIR,
        cache_dir=config.CACHE_DIR,
    )

    print("\n========== [阶段二] 聚类层 / 连拍分组 ==========")
    clusterer = ImageClusterer(
        eps=config.DBSCAN_EPS,
        min_samples=config.DBSCAN_MIN_SAMPLES,
        expected_dim=config.FEATURE_DIM,
    )
    groups = clusterer.run(config.CACHE_DIR)

    if not groups:
        print("🟡 没有发现可比较的连拍组，流程结束。")
        result = {
            "perception": perception_stats,
            "groups": {},
            "ranked_groups": {},
            "llm_decisions": {},
        }
        output_path = _write_results(result)
        print(f"📝 结果已保存: {output_path}")
        return result

    print("\n========== [阶段三] 初筛层 / 审美打分 ==========")
    scorer = AestheticScorer(
        cache_dir=config.CACHE_DIR,
        top_n=top_n,
        centroid_path=config.MASTER_CENTROID_PATH,
        model_path=config.AESTHETIC_MODEL_PATH,
    )
    ranked_groups = scorer.rank_groups(groups)

    print("\n========== [阶段四] 决策层 / 多模态终审 ==========")
    llm_decisions: dict[str, Any] = {}
    should_call_llm = (
        not skip_llm
        and config.LLM_API_KEY
        and config.LLM_BASE_URL
        and config.LLM_MODEL
    )

    if not should_call_llm:
        print("🟡 未配置 LLM API 或传入了 --skip-llm，本轮只输出本地 Top N。")
    else:
        judge = LLMJudge()
        for group_id, candidates in ranked_groups.items():
            candidate_paths = []
            for item in candidates:
                photo_path = _find_photo_path(item["photo_id"])
                if photo_path:
                    candidate_paths.append(photo_path)

            if not candidate_paths:
                print(f"🟡 组 {group_id} 找不到原图文件，跳过 LLM 终审。")
                continue

            decision = judge.judge_candidates(
                candidate_paths=candidate_paths,
                reference_image_path=config.MASTER_REFERENCE_IMAGE,
            )
            llm_decisions[str(group_id)] = decision

    result = {
        "perception": perception_stats,
        "groups": {str(k): v for k, v in groups.items()},
        "ranked_groups": {str(k): v for k, v in ranked_groups.items()},
        "llm_decisions": llm_decisions,
    }
    output_path = _write_results(result)

    print("\n✅ 全流程完成")
    print(f"📝 结果已保存: {output_path}")
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="本地智能摄影选片 Agent")
    parser.add_argument(
        "--skip-llm",
        action="store_true",
        help="跳过多模态大模型终审，只运行本地感知/聚类/评分。",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=config.SCORER_TOP_N,
        help="每个连拍组传给终审的候选数量。",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_pipeline(skip_llm=args.skip_llm, top_n=args.top_n)
