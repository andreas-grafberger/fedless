import glob
import json
import re
from pathlib import Path
from typing import Union

import numpy as np
import pandas as pd
from tensorflow_privacy import get_privacy_spent

GCLOUD_FUNCTION_TIERS = [
    (128, 0.2, 0.000000231),
    (256, 0.4, 0.000000463),
    (512, 0.8, 0.000000925),
    (1024, 1.4, 0.000001650),
    (2048, 2.4, 0.000002900),
    (4096, 4.8, 0.000005800),
    (8192, 4.8, 0.000006800),
]


def calc_gcloud_function_cost(
    memory,
    cpu_ghz,
    invocations,
    function_runtime_seconds,
    function_egress_mb,
    substract_free_tier: bool = True,
):
    INVOCATIONS_UNIT_PRICE = 0.0000004
    GB_SECOND_UNIT_PRICE = 0.0000025
    GHZ_SECOND_UNIT_PRICE = 0.0000100
    NETWORKING_UNIT_PRICE = 0.12

    gb_sec_per_invoc = (memory / 1024.0) * function_runtime_seconds
    ghz_sec_per_invoc = cpu_ghz * function_runtime_seconds
    gb_sec_per_month = gb_sec_per_invoc * invocations
    ghz_sec_per_month = ghz_sec_per_invoc * invocations
    egress_gb_per_month = invocations * function_egress_mb / 1024

    if substract_free_tier:
        invocations = max(0, invocations - 2_000_000)
        gb_sec_per_month = max(0, gb_sec_per_month - 400_000)
        ghz_sec_per_month = max(0, ghz_sec_per_month - 200_000)
        egress_gb_per_month = max(0, egress_gb_per_month - 5)

    return (
        invocations * INVOCATIONS_UNIT_PRICE
        + gb_sec_per_month * GB_SECOND_UNIT_PRICE
        + ghz_sec_per_month * GHZ_SECOND_UNIT_PRICE
        + egress_gb_per_month * NETWORKING_UNIT_PRICE
    )


def read_flower_mnist_log_file(path: Path):
    lines = path.read_text().splitlines()
    progress_lines = [l for l in lines if "fit progress: (" in l]
    entries = []
    round_start_lines = [l for l in lines if "fit_round: strategy sampled" in l]
    first_round_start_time_secs = 0.0
    if len(round_start_lines) > 0:
        first_round_start_time_secs = pd.to_datetime(
            " ".join(round_start_lines[0].split(" ")[2:4])
        ).timestamp()
    for l in progress_lines:
        timestamp = pd.to_datetime(" ".join(l.split(" ")[2:4])).timestamp()
        x = l.split("fit progress: ")[-1]
        round, loss, metrics, time = eval(x)
        # print(first_round_start_time_secs, time - first_round_start_time_secs, time)
        entries.append(
            {
                "round": round,
                "loss": loss,
                "metrics": metrics,
                "accuracy": metrics.get("accuracy"),
                "time-total": time,
                "time": (timestamp - first_round_start_time_secs)
                if len(entries) == 0
                else (time - entries[-1]["time-total"]),
            }
        )
    times_agg_eval = []
    for idx, line in enumerate(lines):
        if idx + 1 == len(lines):
            break
        next_line = lines[idx + 1]
        if "fit_round received" not in line:
            continue
        if "fit progress: (" in next_line:
            t_start = pd.to_datetime(" ".join(line.split(" ")[2:4]))
            t_end = pd.to_datetime(" ".join(next_line.split(" ")[2:4]))
            times_agg_eval.append((t_end - t_start).total_seconds())
    # "time_agg_eval": time_agg_eval,
    df = pd.DataFrame.from_records(entries)
    df["time-aggregation"] = times_agg_eval
    new_dtypes = {
        "time": float,
        "loss": float,
        "accuracy": float,
        "time-total": float,
        "round": int,
        "time-aggregation": float,
    }
    if not df.empty:
        df = df.astype(new_dtypes)
    return df


def read_flower_leaf_log_file(f_err: Path, f_out: Path):
    out_lines = f_out.read_text().splitlines()

    # Metrics
    eval_entries = []
    for idx, l in enumerate(out_lines):
        matches = re.findall(r"EvaluateRes([^\)]+)", l)
        client_accs = []
        client_cardinalities = []
        client_losses = []
        for m in matches:
            eval_dict = eval(f"dict{m})")
            client_acc = eval_dict.get("metrics").get("accuracy")
            client_cardinality = eval_dict.get("num_examples")
            client_loss = eval_dict.get("loss")
            client_accs.append(client_acc)
            client_cardinalities.append(client_cardinality)
            client_losses.append(client_loss)
        if len(matches) == 0:
            continue
        acc = np.average(client_accs, weights=client_cardinalities)
        loss = np.average(client_losses, weights=client_cardinalities)
        eval_entries.append({"round": idx + 1, "accuracy": acc, "loss": loss})
    eval_entries = eval_entries[:-1]
    df = pd.DataFrame.from_records(eval_entries)
    if not df.empty:
        df = df.astype({"round": int, "accuracy": float, "loss": float})

    # Timing info
    err_lines = f_err.read_text().splitlines()
    timing_entries = []
    t_start_training = None
    for idx, line in enumerate(err_lines):
        if "fit_round: strategy sampled" not in line:
            continue
        try:
            received_line = err_lines[idx + 1]
            eval_start_line = err_lines[idx + 2]
            eval_end_line = err_lines[idx + 3]
            assert "evaluate_round received" in eval_end_line
            assert "evaluate_round: strategy sampled" in eval_start_line
            assert "fit_round received" in received_line
        except (IndexError, AssertionError) as e:
            continue
        t_fit_start = pd.to_datetime(" ".join(line.split(" ")[2:4]))
        t_fit_end = pd.to_datetime(" ".join(received_line.split(" ")[2:4]))
        t_eval_start = pd.to_datetime(" ".join(eval_start_line.split(" ")[2:4]))
        t_eval_end = pd.to_datetime(" ".join(eval_end_line.split(" ")[2:4]))
        if not t_start_training:
            t_start_training = t_fit_start
        total_time = (t_eval_end - t_fit_start).total_seconds()
        timing_entries.append(
            {
                "time": total_time,
                "time-eval": (t_eval_end - t_eval_start).total_seconds(),
                "time-aggregation": (t_eval_end - t_fit_end).total_seconds(),
                "time-clients": (t_fit_end - t_fit_start).total_seconds(),
                "time-total": (t_eval_end - t_start_training).total_seconds(),
            }
        )
        # total_seconds
    #    assert len(timing_entries) == len(df)
    timing_df = pd.DataFrame.from_records(timing_entries[: len(df)])

    return df.join(timing_df)


def process_flower_logs(root: Union[str, Path]):
    root = Path(root)
    files = []
    dfs = []

    for f in root.glob("fedless_*.err"):
        if (len(f.name.split("_"))) == 6:  # Local Client Log
            (
                _,
                dataset,
                clients_in_round,
                clients_total,
                local_epochs,
                seed,
            ) = f.name.split("_")
            batch_size = 10
        elif (len(f.name.split("_"))) == 7:  # Local Client Log
            (
                _,
                dataset,
                clients_in_round,
                clients_total,
                local_epochs,
                batch_size,
                seed,
            ) = f.name.split("_")
        else:
            continue
        seed = seed.split(".")[0]
        if dataset == "mnist":  # All required data lies in .err file
            logs_df = read_flower_mnist_log_file(f)
        elif dataset in ["femnist", "shakespeare"]:
            logs_df = read_flower_leaf_log_file(f_err=f, f_out=f.with_suffix(".out"))

        if logs_df.empty:
            continue

        index = pd.MultiIndex.from_tuples(
            [(dataset, clients_in_round, clients_total, local_epochs, batch_size, seed)]
            * len(logs_df),
            names=[
                "dataset",
                "clients-round",
                "clients-total",
                "local-epochs",
                "batch-size",
                "seed",
            ],
        )
        df = pd.DataFrame(
            logs_df.values, index=index, columns=logs_df.columns
        )  # .reset_index()
        df = df.astype(logs_df.dtypes)

        integer_index_levels = [1, 2, 3]
        for i in integer_index_levels:
            df.index = df.index.set_levels(df.index.levels[i].astype(int), level=i)
        dfs.append(df)
    return pd.concat(dfs).sort_index()


def read_fedless_logs(glob_pattern, ignore_dp: bool = True, ignore_flower: bool = True):
    timing_dfs = []
    client_timing_dfs = []
    for folder in glob.glob(glob_pattern):
        folder = Path(folder)
        try:
            tokens = folder.name.split("-")
            if "dp" in tokens:
                if ignore_dp:
                    print(f"Ignoring experiment folder {folder} because ignore_dp=True")
                    continue
                tokens = [t for t in tokens if t != "dp"]
            if "flower" in tokens:
                if ignore_flower:
                    print(
                        f"Ignoring experiment folder {folder} because ignore_flower=True"
                    )
                    continue
                tokens = [t for t in tokens if t != "flower"]
            strategy = tokens[0]
            dataset = tokens[1]
            clients_total = int(tokens[2])
            clients_in_round = int(tokens[3])
            local_epochs = int(tokens[4])
            batch_size = int(tokens[5])
            lr = int(tokens[6])
        except ValueError as e:
            print(e)
            print(f"Error loading {folder}")
            continue
        for timing_file in folder.glob("timing*.csv"):
            seed = timing_file.name.split("_")[1].split(".")[0]
            df = pd.read_csv(timing_file)
            index = pd.MultiIndex.from_tuples(
                [
                    (
                        dataset,
                        clients_in_round,
                        clients_total,
                        local_epochs,
                        batch_size,
                        lr,
                        seed,
                    )
                ]
                * len(df),
                names=[
                    "dataset",
                    "clients-round",
                    "clients-total",
                    "local-epochs",
                    "batch-size",
                    "lr",
                    "seed",
                ],
            )
            df = pd.DataFrame(df.values, index=index, columns=df.columns)
            df.rename(
                columns={
                    "round_id": "round",
                    "global_test_accuracy": "accuracy",
                    "global_test_loss": "loss",
                    "round_seconds": "time",
                    "clients_finished_seconds": "time-clients",
                    "aggregator_seconds": "time-aggregation",
                    "num_clients_round": "clients",
                },
                inplace=True,
            )
            new_dtypes = {
                "session_id": str,
                "round": int,
                "accuracy": float,
                "loss": float,
                "time": float,
                "time-clients": float,
                "time-aggregation": float,
                "clients": int,
            }
            if not df.empty:
                df = df.astype(new_dtypes)
            timing_dfs.append(df)
        for client_file in folder.glob("clients*.csv"):
            seed = client_file.name.split("_")[1].split(".")[0]
            df = pd.read_csv(client_file)
            index = pd.MultiIndex.from_tuples(
                [
                    (
                        dataset,
                        int(clients_in_round),
                        int(clients_total),
                        int(local_epochs),
                        int(batch_size),
                        lr,
                        seed,
                    )
                ]
                * len(df),
                names=[
                    "dataset",
                    "clients-round",
                    "clients-total",
                    "local-epochs",
                    "batch-size",
                    "lr",
                    "seed",
                ],
            )
            df = pd.DataFrame(df.values, index=index, columns=df.columns)
            new_dtypes = {"seconds": float, "round": int}
            if not df.empty:
                df = df.astype(new_dtypes)
            client_timing_dfs.append(df)

    timing_df = pd.concat(timing_dfs).sort_index()
    client_df = pd.concat(client_timing_dfs).sort_index()

    # Add column with FaaS platform name
    client_df["state"] = client_df["round"].map(lambda r: "cold" if (r < 1) else "warm")
    client_df["function"] = client_df["function"].map(lambda x: json.loads(x))
    client_df["platform"] = client_df["function"].map(
        lambda x: x["type"]
        if x["type"] != "openwhisk-web"
        else ("ibm" if "ibm" in x["params"]["endpoint"] else "lrz")
    )

    return timing_df, client_df


def read_privacy_simulation_results(result_dir: Path):
    parameters = [
        "devices",
        "epochs",
        "local-epochs",
        "local-batch-size",
        "clients-round",
        "l2-norm",
        "noise-multiplier",
        "ldp",
        "microbatches",
        "time-start",
    ]

    dfs = []
    for f in result_dir.glob("results_100_*_5_16_25_*.csv"):

        parameter_values = f.stem.lstrip("results_").split("_")
        (
            devices,
            epochs,
            local_epochs,
            local_batch_size,
            clients_round,
            l2_norm,
            noise_multiplier,
            ldp,
            microbatches,
            time_start,
        ) = parameter_values

        assert len(parameter_values) == len(parameters)

        df = pd.read_csv(f)
        index = pd.MultiIndex.from_tuples(
            [
                (
                    local_batch_size,
                    l2_norm,
                    noise_multiplier,
                    ldp,
                    microbatches,
                    time_start,
                )
            ]
            * len(df),
            names=[
                "local-batch-size",
                "l2-norm",
                "noise-multiplier",
                "ldp",
                "microbatches",
                "time-start",
            ],
        )
        df = pd.DataFrame(df.values, index=index, columns=df.columns)
        df = df.reset_index()
        df["ldp"] = df["ldp"].apply(eval)
        df = df.rename(
            columns={
                "local_epochs": "local-epochs",
                "clients_per_round": "clients-round",
                "clients_call_duration": "time-clients",
                "clients_histories": "clients-histories",
                "privacy_params": "privacy-params",
                "privacy_guarantees": "privacy-guarantees",
                "test_loss": "loss",
                "test_accuracy": "accuracy",
            }
        )
        new_dtypes = {
            "devices": int,
            "epochs": int,
            "local-epochs": int,
            "local-batch-size": int,
            "clients-round": int,
            "l2-norm": float,
            "noise-multiplier": float,
            "ldp": bool,
            "microbatches": int,
            "time-start": str,
            "time-clients": float,
            "accuracy": float,
            "loss": float,
        }
        if not df.empty:
            df = df.astype(new_dtypes)
        df[df["microbatches"] == 0]["microbatches"] = "16"
        df = df.set_index(
            ["local-batch-size", "ldp", "microbatches", "time-start", "epoch"]
        )
        dfs.append(df)

    df = pd.concat(dfs).sort_index()

    # Extract and format privacy parameters
    df["privacy-params"] = df[["l2-norm", "noise-multiplier"]].apply(
        lambda x: f'{x["l2-norm"]}, {x["noise-multiplier"]}'
        if all(x.notnull())
        else None,
        axis=1,
    )
    priv_guarant_deser = (
        df["privacy-guarantees"].map(lambda x: x.replace("'", "")).map(json.loads)
    )
    priv_guarant_deser_dict = priv_guarant_deser.map(lambda x: x[0] if x else None)
    df["eps-round-clients"] = priv_guarant_deser_dict.map(
        lambda x: x["eps"] if x and "eps" in x else None
    )
    df["delta-round-clients"] = priv_guarant_deser_dict.map(
        lambda x: x["delta"] if x and "eps" in x else None
    )
    df["eps-delta-round-client"] = df[
        ["eps-round-clients", "delta-round-clients"]
    ].apply(
        lambda x: f'({x["eps-round-clients"]}, {x["delta-round-clients"]})'
        if all(x.notnull())
        else None,
        axis=1,
    )

    # RDP Accountant
    client_pick_prob = df["clients-round"] / df["devices"]
    df["rdp-round-clients"] = priv_guarant_deser_dict.map(
        lambda x: x["rdp"] if x and "rdp" in x else None
    ).apply(np.asarray)
    df["orders-round-clients"] = priv_guarant_deser_dict.map(
        lambda x: x["orders"] if x and "orders" in x else None
    ).apply(np.asarray)
    df["rdp-cumsum"] = df.groupby("time-start")["rdp-round-clients"].transform(
        pd.Series.cumsum
    )
    df["cum-privacy"] = df[
        ["orders-round-clients", "rdp-cumsum", "delta-round-clients"]
    ].apply(
        lambda row: get_privacy_spent(row[0], row[1], target_delta=row[2])
        if all(row.notnull())
        else None,
        axis=1,
    )
    df["cum-eps"] = df["cum-privacy"].apply(lambda x: x[0] if x else None)
    df["cum-eps-div"] = df["cum-eps"] * client_pick_prob

    # Naive Accounting
    df["cum-eps-naive"] = df.groupby("time-start")["eps-round-clients"].transform(
        pd.Series.cumsum
    )
    df["cum-eps-naive-div"] = df["cum-eps-naive"] * client_pick_prob
    df["delta-round-clients-cumsum"] = df.groupby("time-start")[
        "delta-round-clients"
    ].transform(pd.Series.cumsum)
    df["delta-round-clients-cumsum-div"] = (
        df["delta-round-clients-cumsum"] * client_pick_prob
    )

    return (
        df.reset_index()
        .set_index(["ldp", "microbatches", "l2-norm", "noise-multiplier"])
        .sort_index()
    )
