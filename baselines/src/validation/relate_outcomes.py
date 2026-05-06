from scipy.stats import linregress, pearsonr, spearmanr
from omegaconf import DictConfig
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib as mpl
import numpy as np
import os
from pathlib import Path
from statsmodels.api import OLS, add_constant
import seaborn as sns
from hydra.utils import log
import hydra
from sklearn.model_selection import LeaveOneOut
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error


def get_slope(session: pd.DataFrame, normalize: bool=True):
    client_utt = session[session["speaker"] == "client"]
    y = client_utt['score'].to_numpy()
    if len(y) < 2:
        return None
    x = 100 * session.index.get_indexer(client_utt.index) / (len(session) - 1) if normalize else np.arange(len(y))
    result = linregress(x, y)
    return result

def get_confidence_scores(prolific_id: str, delta_with: str) -> tuple:
    ruler_df = pd.read_csv("data/2024-11-14-MIV6.3A-2024-11-22-MIV6.3A_all_data_delta_with_post_keep_high_conf_True_merged.csv")
    match = ruler_df[ruler_df["Participant id"] == prolific_id]
    if match.empty:
        return None, None, None  # No match found
    pre = match["pre_confidence"].values[0]
    post = match[f"{delta_with}_confidence"].values[0]
    delta = post - pre
    return pre, post, delta

def compute_row(group, delta_with="week_later"):
    pid = group['conv_id'].iloc[0]
    result = get_slope(group, normalize=True)
    pre, post, delta = get_confidence_scores(pid, delta_with)

    c = (group["t1_label_auto"] == "C").sum()
    s = (group["t1_label_auto"] == "S").sum()
    r = group["t2_label_auto"].isin(["SR", "CR"]).sum()
    q = (group["t1_label_auto"] == "Q").sum()

    mico_codes = ['AF', 'ADP', 'EC', 'RCP', 'SU', 'OQ', 'CR', 'SR']
    miin_codes = ['ADW', 'CO', 'DI', 'RCW', 'WA']
    mico = group["t2_label_auto"].isin(mico_codes).sum()
    miin = group["t2_label_auto"].isin(miin_codes).sum()

    percent_mico = mico / (mico + miin) if (mico + miin) > 0 else 0
    rq = r / q if q > 0 else 0
    percent_ct = c / (c + s) if (c + s) > 0 else 0

    return pd.Series({
        "conv_id": pid,
        'pre_confidence': pre,
        f'{delta_with}_confidence': post,
        f"delta_confidence": delta,
        "slope": result.slope,
        "intercept": result.intercept,
        "r_value": result.rvalue,
        "%MIC": percent_mico,
        "R:Q": rq,
        "%CT": percent_ct,
    })

def linear_regression(df, delta_with="week_later"):
    results = (
        df.groupby("conv_id", sort=False, group_keys=False)
        .apply(lambda group: compute_row(group, delta_with=delta_with))
        .dropna(subset=["delta_confidence"])
    )

    target = "delta_confidence"
    exclude = {f"{delta_with}_confidence", "conv_id", "r_value", "intercept"}
    
    corr_results = []
    for col in results.columns:
        if col in exclude or col == target:
            continue
        if pd.api.types.is_numeric_dtype(results[col]):
            x = results[target]
            y = results[col]
            if x.isna().any() or y.isna().any():
                continue  # skip columns with missing data
            # r, p = pearsonr(x, y)
            r, p = spearmanr(x, y)
            corr_results.append((col, r, p))
    corr_df = pd.DataFrame(corr_results, columns=["variable", "spearman_r", "p_value"])

    # Optional: return both the results and the correlations
    exp_output_dir = Path(hydra.core.hydra_config.HydraConfig.get().runtime.output_dir)
    os.makedirs(exp_output_dir / "results", exist_ok=True)

    corr_df.to_csv(exp_output_dir / 'results/correlation_results.csv', index=False)
    results.to_csv(exp_output_dir / 'results/agg.csv')
    log.info(f"Saved aggregated results to {exp_output_dir / 'results/agg.csv'}")
    log.info(f"Saved correlation results to {exp_output_dir / 'results/correlation_results.csv'}")

    # r, p = pearsonr(results['slope'], results['delta_confidence'])
    r, p = spearmanr(results['slope'], results['delta_confidence'])
    # print(f"Pearson correlation between slope and delta: r = {r:.2f}, p = {p:.4f}")

    X = add_constant(results['slope'])
    y = results['delta_confidence']
    model = OLS(y, X).fit()
    print(model.summary())
    plt.figure(figsize=(6, 4))
    plt.scatter(results['slope'], results['delta_confidence'], alpha=0.8)
    pred_line = model.predict(X)
    plt.plot(results['slope'], pred_line, 'r--', label=f'Regression Line\nr={r:.2f}, p={p:.4f}')
    plt.xlabel('Motivation Slope')
    plt.ylabel(f'Change in Confidence ({delta_with})')
    # plt.title(f'Motivation Trend vs. Change in  Gain (n={len(results)}, delta with {delta_with})')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(exp_output_dir / 'results/motivation_vs_confidence.pdf', format="pdf", bbox_inches='tight')
    plt.show()
    log.info(f"Saved motivation vs confidence plot to {exp_output_dir / 'results/motivation_vs_confidence.pdf'}")

    return results, corr_df

def loocv(df, delta_with="week_later"):
    results = (
        df.groupby("conv_id", sort=False, group_keys=False)
        .apply(lambda group: compute_row(group, delta_with=delta_with))
        .dropna(subset=["delta_confidence"])
    )
    # features = ['slope', 'pre_confidence', '%MIC', 'R:Q', '%CT']
    features = ['slope']
    X = results[features].values
    y = results['delta_confidence'].values
    conv_ids = results['conv_id'].values

    loo = LeaveOneOut()
    y_pred, y_true, test_ids = [], [], []

    print("\n=== LOOCV PREDICTION SUMMARY ===")
    for train_idx, test_idx in loo.split(X):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        model = LinearRegression()
        model.fit(X_train, y_train)
        y_hat = model.predict(X_test)[0]
        y_pred.append(y_hat)
        y_true.append(y_test[0])
        test_ids.append(conv_ids[test_idx][0])
        # print(f"Test PID: {conv_ids[test_idx][0]:<10} | slope: {X[test_idx][0][0]:>6.3f} | "
        #       f"true Δconf: {y_test[0]:>+5.2f} | predicted: {y_hat:>+5.2f} | error: {y_hat - y_test[0]:>+5.2f}")

    r2 = r2_score(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    print(f"\nFinal R²: {r2:.3f}, MSE: {mse:.3f}\n")

    # Plot predictions vs true values
    plt.figure(figsize=(6, 4))
    plt.scatter(y_true, y_pred, alpha=0.7)
    plt.plot([min(y_true), max(y_true)], [min(y_true), max(y_true)], 'r--', label="Ideal Fit")
    plt.xlabel("True Δ Confidence")
    plt.ylabel("Predicted Δ Confidence")
    plt.title(f"LOOCV: slope → Δ Confidence\nn={len(y)}, R²={r2:.2f}, MSE={mse:.2f}")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

    err = np.array(y_pred) - np.array(y_true)
    with np.errstate(divide='ignore', invalid='ignore'):
        percent_err = np.where(np.abs(y_true) > 1e-5, (err / np.array(y_true)) * 100, np.nan)


    res = pd.DataFrame({
        "conv_id": test_ids,
        **dict(zip(features, X.T)),
        "true_delta": y_true,
        "predicted_delta": y_pred,
        "error": err,
        "percent_error": percent_err,
    })
    exp_output_dir = Path(hydra.core.hydra_config.HydraConfig.get().runtime.output_dir)
    res.to_csv(exp_output_dir / 'results/loocv_results.csv', index=False)
    log.info(f"Saved LOOCV results to {exp_output_dir / 'results/loocv_results.csv'}")
    return res


def plot_trajectory(df: pd.DataFrame, results: pd.DataFrame, pids: str, normalize: bool=True, delta_with: str="week_later"):
    
    plt.figure(figsize=(10, 4))
    for i, pid in enumerate(pids):
        session = df[df['conv_id'] == pid]
        client_utt = session[session["speaker"] == "client"]
        y = client_utt['score'].to_numpy()
        x = 100 * session.index.get_indexer(client_utt.index) / (len(session) - 1) if normalize else np.arange(len(y))

        session_result = results[results['conv_id'] == pid]
        slope = session_result['slope'].iloc[0]
        intercept = session_result['intercept'].iloc[0]
        r_value = session_result['r_value'].iloc[0]
        pre = session_result['pre_confidence'].iloc[0]
        post = session_result[f'{delta_with}_confidence'].iloc[0]
        delta = session_result['delta_confidence'].iloc[0]
        plt.plot(x, y, 'o-', label = f"ID{i+1}: Δconf={int(delta):+} ({int(pre)}→{int(post)}), slope={slope:+.2f}")

        line = intercept + slope * np.array(x)
        color = 'green' if slope > 0 else 'red'
        plt.plot(x, line, '--', color=color, linewidth=2)

    plt.xlabel('% Session Progress' if normalize else 'Utterance Index', fontsize=14)
    plt.ylabel('Motivation Score', fontsize=14)
    plt.ylim(-2.2,2.2)
    plt.gca().yaxis.set_major_locator(ticker.MultipleLocator(1))
    plt.grid(True)
    plt.legend(fontsize=12)
    plt.tight_layout()
    exp_output_dir = Path(hydra.core.hydra_config.HydraConfig.get().runtime.output_dir)
    plt.savefig(exp_output_dir / 'results/sample_traj.pdf', format="pdf", bbox_inches='tight')
    plt.show()
    log.info(f"Saved sample trajectories plot to {exp_output_dir / 'results/sample_traj.pdf'}")

def relate_outcomes(cfg: DictConfig) -> None:
    mpl.rcParams['font.family'] = 'Times New Roman'

    auto_anno_path = Path('data/annotated') / (
        f"{cfg.input_dataset.name}_"
        f"{cfg.input_dataset.subset}_"
        f"{cfg.annotated.class_structure}_"
        f"{cfg.annotated.model.rsplit('/', 1)[-1]}_"
        f"{cfg.annotated.context_mode}_"
        f"{cfg.annotated.num_context_turns if cfg.annotated.context_mode == 'interval' else ''}" 
        f"_annotated.csv"
        # f"_annotated_old.csv"
    )
    log.info(f"AutoMISC annotations path: {auto_anno_path}")
    df = pd.read_csv(auto_anno_path)
    label_to_score = {
        "TS-": -2, "AC-": -2, "C-": -2, 
        "N-": -1, "R-": -1, "AB-": -1, "D-": -1, 
        "O-": 0,
        "N": 0, 
        "O+": 0,
        "D+": 1, "AB+": 1, "R+": 1, "N+": 1, 
        "C+": 2, "AC+": 2, "TS+": 2, 
    }
    df['score'] = df['t2_label_auto'].map(label_to_score)

    results, _ = linear_regression(df, delta_with="week_later")
    loocv(df, delta_with="week_later")

    ids_to_plot = ['6615283f82799fd70e2c056b', 
                   '5e19b24b2bf9512392d6dd90',
                   ]
    plot_trajectory(df, results, ids_to_plot, normalize=True, delta_with="week_later")

