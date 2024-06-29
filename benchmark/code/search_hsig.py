import argparse
import os
import time
from typing import List

import h5py
import numpy as np
import pandas as pd
import utils
from alg_base import BaseANN
from alg_hannlib import HannLib
from alg_hnswlib import HnswLib


class SearchStategy:
    HYBRID_FITERING = 0
    PRE_FILTERING = 1
    POST_FILTERING = 2
    CBO = 3


def read_hdf5_dataset(filepath, keys: List[str]):
    with h5py.File(filepath, "r") as f:
        ret = []
        for k in keys:
            ret.append(f[k][:])
    return ret


def bench_hybrid_query(
    index: BaseANN,
    search_strategy: int,
    optimizer_conf_dir,
    k,
    ef_list,
    al_list,
    low_range,
    high_range,
    query_vectors,
    query_ranges,
    gt,
):
    print("Searching index...")
    time_list = []
    recall_list = []

    base_params = {
        "search_strategy": search_strategy,
        "optimizer_conf_dir": optimizer_conf_dir,
        "target_recall": 0.9,
        "ef_factor": 1,
        "low_range": low_range,
        "high_range": high_range,
        "ef": k,
        "al": 16,
    }

    if search_strategy != SearchStategy.CBO:
        ef_data = []
        al_data = []
        for ef in ef_list:
            for al in al_list:
                ef_data.append(ef)
                al_data.append(al)
                base_params["ef"] = ef
                base_params["al"] = al
                index.set_query_arguments(base_params)
                total_time = 0
                i = 0
                results = np.zeros(
                    [len(query_vectors), k], dtype="int64"
                )  

                for q, r in zip(query_vectors, query_ranges):
                    if i % 100 == 0:
                        print(f"Executiong query {i} under ef={ef}, al={al}...")
                    start = time.time()
                    I = index.hybrid_query(q, r, k)
                    end = time.time()
                    total_time += end - start
                    results[i, : len(I)] = I
                    i += 1
                recall = utils.compute_recall(results, gt, k, k)
                avg_time = total_time / len(query_ranges) * 1000
                time_list.append(avg_time)
                recall_list.append(recall)

        df = pd.DataFrame(
            {
                "ef": ef_data,
                "al": al_data,
                "recall": recall_list,
                "latency(ms)": time_list,
            }
        )
        df["QPS"] = 1000 / df["latency(ms)"]
    else:
        ef_data = []
        al_data = []
        for ef in ef_list:
            for al in al_list:
                ef_data.append(ef)
                al_data.append(al)
                base_params["ef"] = ef
                base_params["al"] = al
                base_params["low_range"] = low_range
                base_params["high_range"] = high_range
                index.set_query_arguments(base_params)
                total_time = 0
                i = 0
                results = np.zeros(
                    [len(query_vectors), k], dtype="int64"
                )  # 存储每个query_vector的knn

                for q, r in zip(query_vectors, query_ranges):
                    if i % 100 == 0:
                        print(
                            f"Executiong query {i} under ef={ef}, al={al}..."
                        )
                    start = time.time()
                    I = index.hybrid_query(q, r, k)
                    end = time.time()
                    total_time += end - start
                    results[i, : len(I)] = I
                    i += 1
                recall = utils.compute_recall(results, gt, k, k)
                avg_time = total_time / len(query_ranges) * 1000
                time_list.append(avg_time)
                recall_list.append(recall)
        df = pd.DataFrame(
            {
                "ef": ef_data,
                "al": al_data,
                "recall": recall_list,
                "latency(ms)": time_list,
            }
        )
        df["QPS"] = 1000 / df["latency(ms)"]
    return df


def build_or_load_index(name: str, params, base, base_scalars, index_save_path):
    index = None
    if name == "HNSW":
        index = HnswLib(metric="euclidean", method_param=params)
    else:
        index = HannLib(metric="euclidean", method_param=params)
    
    if os.path.exists(index_save_path):
        print(f"Reading index from {index_save_path} ...")
        start, end = 0, 0
        index.loadIndex(base, index_save_path)
        index.scalars = base_scalars
    else:
        print(f"Building index: {index.name}...")
        start = time.time()
        index.fit(base, base_scalars)
        end = time.time()
        index.saveIndex(index_save_path)

        print(f"Index built: {index.name}, duration: {end-start}.")
        with open(index_save_path + ".time", "w") as f:
            f.write(f"{end - start}")

    return index, end - start


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Index parameters")
    parser.add_argument("--k", type=int, default=10, help="For kNN search")
    parser.add_argument(
        "--n_query_to_use", type=int, default=1000, help="Number of queries to use"
    )
    parser.add_argument(
        "--data_path", type=str, required=True, help="Path to the hdf5 data file"
    )
    parser.add_argument(
        "--use_mbv_hnsw", type=bool, default=False, help="Whether to use MBV-HNSW index"
    )
    parser.add_argument("--M", type=int, default=16, help="Number of graph connections")
    parser.add_argument(
        "--efConstruction", type=int, default=500, help="Parameter for HNSW index"
    )
    parser.add_argument("--B", type=int, default=8, help="Number of buckets")
    parser.add_argument("--low_range", type=float, default=0.1, help="low query range")
    parser.add_argument("--high_range", type=float, default=0.5, help="high query range")
    parser.add_argument(
        "--index_cache_path",
        type=str,
        required=True,
        help="Directory to load and save the index",
    )
    parser.add_argument(
        "--optimizer_conf_dir",
        type=str,
        required=False,
        help="Directory of optimizer configuration files, only effective if rso is enabled",
    )
    parser.add_argument(
        "--plan",
        type=int,
        default=0,
        help="Search plan: 0 (hybrid filtering), 1 (pre fitering), 2 (post filtering), 3 (CBO)",
    )
    parser.add_argument(
        "--target_recall_list",
        type=float,
        nargs="+",
        default=[0.9, 0.91, 0.92, 0.93, 0.94, 0.95, 0.96, 0.97, 0.98, 0.99, 1],
        help="List of target recall values, only effective if rso is enabled",
    )
    parser.add_argument(
        "--ef_factor_list",
        type=float,
        nargs="+",
        default=[0.8, 1.0, 2.0],
        help="List of ef_factor values, only effective if rso is enabled",
    )
    parser.add_argument(
        "--ef_list",
        type=int,
        nargs="+",
        default=list(range(10, 2000, 20)),
        help="List of EF values",
    )
    parser.add_argument(
        "--al_list", type=int, nargs="+", default=[8, 16, 32], help="List of AL values"
    )
    parser.add_argument(
        "--result_save_path",
        type=str,
        required=True,
        help="File path to save the benchmark results",
    )

    args = parser.parse_args()

    if os.path.exists(args.result_save_path):
        print(f"Result file exists, skip: {args.result_save_path}")
        exit(0)

    if args.use_mbv_hnsw:
        name = "MBV-HNSW"
        params = {
            "num_slots": args.B,
            "M": args.M,
            "efConstruction": args.efConstruction,
        }
    else:
        name = "HNSW"
        params = {"M": args.M, "efConstruction": args.efConstruction}

    (
        base,
        base_scalars,
        test,
        test_ranges,
        test_hybrid_knn,
    ) = read_hdf5_dataset(
        args.data_path,
        ["base", "base_scalars", "test", "test_ranges", "test_hybrid_knn"],
    )
    index, _ = build_or_load_index(
        name, params, base, base_scalars, args.index_cache_path
    )

    nq = args.n_query_to_use
    results = bench_hybrid_query(
        index,
        args.plan,
        args.optimizer_conf_dir,
        args.k,
        args.ef_list,
        args.al_list,
        args.low_range,
        args.high_range,
        test[:nq],
        test_ranges[:nq],
        test_hybrid_knn[:nq],
    )
    results.to_csv(args.result_save_path, index=False)
    print(results)
    print(f"Results were saved to {args.result_save_path}")
