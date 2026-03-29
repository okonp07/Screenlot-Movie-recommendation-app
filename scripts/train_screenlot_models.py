from __future__ import annotations

import argparse
from pathlib import Path
import sys
from time import time

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from screenlot.modeling import (  # noqa: E402
    BENCHMARK_JSON,
    TRAINED_HYBRID_ARTIFACT,
    ContentSimilarityModel,
    ItemKNNRecommender,
    MatrixFactorizationRecommender,
    PopularityRecommender,
    TrainedScreenLotHybridRecommender,
    benchmark_model,
    filter_active_entities,
    leave_last_two_split,
    load_processed_catalog,
    load_ratings_frame,
    sample_user_ids_for_target_rows,
    sample_holdout_rows,
    sample_users,
    select_best_model_name,
    build_user_profiles,
    save_benchmarks,
    save_model_card,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train and benchmark ScreenLot recommendation models.")
    parser.add_argument(
        "--max-rows",
        type=int,
        default=1_500_000,
        help="Approximate target rating volume for a representative, user-complete training sample. Use 0 for the full dataset.",
    )
    parser.add_argument("--max-users", type=int, default=12_000, help="Optional cap on distinct users after filtering. Use 0 for all.")
    parser.add_argument("--min-user-interactions", type=int, default=5)
    parser.add_argument("--min-item-interactions", type=int, default=20)
    parser.add_argument("--max-eval-users", type=int, default=1000, help="Maximum users to benchmark on the validation/test holdouts.")
    parser.add_argument("--item-knn-neighbors", type=int, default=120)
    parser.add_argument("--mf-components", type=int, default=64)
    parser.add_argument("--positive-threshold", type=float, default=3.5)
    parser.add_argument("--artifact-path", type=Path, default=TRAINED_HYBRID_ARTIFACT)
    parser.add_argument("--benchmark-path", type=Path, default=BENCHMARK_JSON)
    args = parser.parse_args()

    started = time()
    print("Loading processed catalog...")
    catalog = load_processed_catalog()

    print("Loading ratings...")
    if args.max_rows > 0:
        sampled_user_ids = sample_user_ids_for_target_rows(
            target_rows=args.max_rows,
            min_user_interactions=args.min_user_interactions,
        )
        print(
            f"Selected {len(sampled_user_ids):,} users for an approximately "
            f"{args.max_rows:,}-rating training sample."
        )
        ratings = load_ratings_frame(sampled_user_ids=sampled_user_ids)
    else:
        ratings = load_ratings_frame()
    print(f"Loaded {len(ratings):,} ratings across {ratings['userId'].nunique():,} users.")

    print("Filtering active users and items...")
    ratings = filter_active_entities(
        ratings,
        min_user_interactions=args.min_user_interactions,
        min_item_interactions=args.min_item_interactions,
    )
    ratings = sample_users(ratings, max_users=args.max_users)
    ratings = filter_active_entities(
        ratings,
        min_user_interactions=args.min_user_interactions,
        min_item_interactions=args.min_item_interactions,
    )
    print(f"Training subset: {len(ratings):,} ratings, {ratings['userId'].nunique():,} users, {ratings['movieId'].nunique():,} items.")

    catalog = catalog[catalog["movie_id"].isin(ratings["movieId"].unique())].copy()
    print(f"Model catalog subset: {len(catalog):,} movies.")

    splits = leave_last_two_split(ratings, min_user_interactions=args.min_user_interactions)
    calibration_rows = splits.validation.copy()
    test_rows = sample_holdout_rows(splits.test, max_users=args.max_eval_users)
    user_profiles = build_user_profiles(splits.train, positive_threshold=args.positive_threshold)
    print(f"Train/validation/test sizes: {len(splits.train):,} / {len(calibration_rows):,} / {len(test_rows):,}")

    print("Fitting popularity baseline...")
    popularity = PopularityRecommender().fit(splits.train)

    print("Fitting content model...")
    content = ContentSimilarityModel().fit(catalog)

    print("Fitting item-item collaborative model...")
    item_knn = ItemKNNRecommender(
        n_neighbors=args.item_knn_neighbors,
        positive_threshold=args.positive_threshold,
    ).fit(splits.train)

    print("Fitting matrix factorization baseline...")
    matrix_factorization = MatrixFactorizationRecommender(
        n_components=args.mf_components,
        positive_threshold=max(args.positive_threshold - 0.5, 2.5),
    ).fit(splits.train)

    print("Training hybrid reranker on validation holdout...")
    hybrid = TrainedScreenLotHybridRecommender(
        catalog=catalog,
        popularity_model=popularity,
        content_model=content,
        item_knn_model=item_knn,
        matrix_factorization_model=matrix_factorization,
    ).fit_calibrator(calibration_rows, user_profiles)

    raw_hybrid_test = benchmark_model(
        hybrid,
        test_rows,
        user_profiles,
        catalog,
        popularity.popularity_probabilities,
        model_name="hybrid",
    )

    print("Selecting the ScreenLot serving model on validation holdout...")
    validation_benchmark_rows = [
        benchmark_model(popularity, calibration_rows, user_profiles, catalog, popularity.popularity_probabilities),
        benchmark_model(item_knn, calibration_rows, user_profiles, catalog, popularity.popularity_probabilities),
        benchmark_model(matrix_factorization, calibration_rows, user_profiles, catalog, popularity.popularity_probabilities),
        benchmark_model(hybrid, calibration_rows, user_profiles, catalog, popularity.popularity_probabilities),
    ]
    selected_model_name = select_best_model_name(validation_benchmark_rows)
    hybrid.configure_serving_model(selected_model_name, validation_benchmarks=validation_benchmark_rows)
    print(f"Validation-selected serving model: {hybrid.model_label()}")

    print("Benchmarking models on test holdout...")
    benchmark_rows = [
        benchmark_model(popularity, test_rows, user_profiles, catalog, popularity.popularity_probabilities),
        benchmark_model(item_knn, test_rows, user_profiles, catalog, popularity.popularity_probabilities),
        benchmark_model(matrix_factorization, test_rows, user_profiles, catalog, popularity.popularity_probabilities),
        raw_hybrid_test,
        benchmark_model(
            hybrid,
            test_rows,
            user_profiles,
            catalog,
            popularity.popularity_probabilities,
            model_name="screenlot_serving",
        ),
    ]
    save_benchmarks(
        benchmark_rows,
        benchmark_json_path=args.benchmark_path,
        benchmark_csv_path=args.benchmark_path.with_suffix(".csv"),
    )

    artifact_path = hybrid.save(args.artifact_path)
    config = {
        "max_rows": args.max_rows,
        "max_users": args.max_users,
        "min_user_interactions": args.min_user_interactions,
        "min_item_interactions": args.min_item_interactions,
        "max_eval_users": args.max_eval_users,
        "item_knn_neighbors": args.item_knn_neighbors,
        "mf_components": args.mf_components,
        "positive_threshold": args.positive_threshold,
        "elapsed_seconds": round(time() - started, 2),
    }
    save_model_card(
        benchmark_rows,
        artifact_path,
        config,
        validation_benchmarks=validation_benchmark_rows,
        selected_model_name=selected_model_name,
        hybrid_details={
            "score_strategy": hybrid.score_strategy,
            "blend_weights": hybrid.blend_weights,
            "model_label": hybrid.model_label(),
        },
    )

    benchmark_frame = pd.DataFrame(benchmark_rows)
    print("\nScreenLot benchmark summary")
    print(benchmark_frame.to_string(index=False))
    print(f"\nHybrid artifact: {artifact_path}")
    print(f"Benchmark JSON: {args.benchmark_path}")


if __name__ == "__main__":
    main()
