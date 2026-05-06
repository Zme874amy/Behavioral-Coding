import os
from pathlib import Path
from omegaconf import DictConfig
import pandas as pd
from sklearn.metrics import confusion_matrix, cohen_kappa_score, classification_report
import numpy as np
from scipy.stats import norm
import seaborn as sns
import matplotlib.pyplot as plt
import hydra
from hydra.utils import log

manual_fn = 'MIV6.3A_manual.csv' 

def compute_p0_pe(y1, y2):
    cm = confusion_matrix(y1, y2)
    total = np.sum(cm)
    p0 = np.trace(cm) / total
    row_marginals = np.sum(cm, axis=1) / total
    col_marginals = np.sum(cm, axis=0) / total
    pe = np.sum(row_marginals * col_marginals)
    return p0, pe, cm, row_marginals, col_marginals

def compute_asymptotic_variance_kappa(p0, pe, cm, row_marginals, col_marginals, n):
    theta_1 = p0  # Observed agreement
    theta_2 = pe  # Expected agreement
    total = np.sum(cm)
    p_ij = cm / total
    theta_3 = np.sum(np.diag(p_ij) * (row_marginals + col_marginals))
    theta_4 = np.sum([
        p_ij[i, j] * (row_marginals[i] + col_marginals[j]) ** 2
        for i in range(p_ij.shape[0])
        for j in range(p_ij.shape[1])
    ])
    var_kappa = (1 / n) * (
        (theta_1 * (1 - theta_1)) / (1 - theta_2) ** 2 +
        (2 * (1 - theta_1) * (2 * theta_1 * theta_2 - theta_3)) / (1 - theta_2) ** 3 +
        ((1 - theta_1) ** 2 * (theta_4 - 4 * theta_2 ** 2)) / (1 - theta_2) ** 4
    )
    return var_kappa

def compute_p_value(kappa, var_kappa):
    z_score = kappa / np.sqrt(var_kappa)
    log.info(f"Z-score: {z_score}")
    p_value = 2 * (1 - norm.cdf(abs(z_score)))
    return p_value

def cohens_kappa(cfg, auto_anno_path, manual_path, tier, rater):
    exp_output_dir = Path(hydra.core.hydra_config.HydraConfig.get().runtime.output_dir)
    os.makedirs(exp_output_dir / "results", exist_ok=True)

    auto_df = pd.read_csv(auto_anno_path)
    manual_df = pd.read_csv(manual_path)
    # data = pd.read_csv(file_path, nrows=n)

    client_codes_map = {
        'T1': ["C", "S", "N"],
        'T2': ["TS-", "AC-", "C-", "N-", "R-", "AB-", "D-", "O-", "N", "O+", "D+", "AB+", "R+", "N+", "C+", "AC+", "TS+"]
    }
    counsellor_code_map = {
        'T1': ["CRL", "SRL", "IMC", "IMI", "Q", "O"],
        'T2': ["CR", "AF", "SU", "RF", "EC", "SR", "ADP", "RCP", "GI", "ADW", "CO", "DI", "RCW", "WA", "OQ", "CQ", "FA", "FI", "ST"]
    }
        
    counsellor_codes = counsellor_code_map[tier]
    client_codes = client_codes_map[tier]

    human_labels = manual_df[f'{tier.lower()}_label_{rater}']
    automisc_labels = auto_df[f'{tier.lower()}_label_auto']
    filtered_human_labels = human_labels[human_labels.isin(counsellor_codes + client_codes)]
    filtered_automisc_labels = automisc_labels[automisc_labels.isin(counsellor_codes + client_codes)]

    # Ensure both series are aligned
    filtered_human_labels = filtered_human_labels[filtered_human_labels.index.isin(filtered_automisc_labels.index)]
    filtered_automisc_labels = filtered_automisc_labels[filtered_automisc_labels.index.isin(filtered_human_labels.index)]

    # --------------------- Counsellor Calculations ---------------------
    counsellor_human_labels = filtered_human_labels[filtered_human_labels.isin(counsellor_codes)]
    counsellor_automisc_labels = filtered_automisc_labels[filtered_automisc_labels.isin(counsellor_codes)]

    co_po, co_pe, cm_co, row_marginals_co, col_marginals_co = compute_p0_pe(counsellor_human_labels, counsellor_automisc_labels)
    man_co = (co_po - co_pe) / (1 - co_pe)

    kappa_score_counsellor = cohen_kappa_score(counsellor_human_labels, counsellor_automisc_labels)
    av_counsellor = compute_asymptotic_variance_kappa(co_po, co_pe, cm_co, row_marginals_co, col_marginals_co, len(counsellor_human_labels))
    p_co = compute_p_value(kappa_score_counsellor, av_counsellor)

    log.info(f"Cohen's Kappa ({tier} Counsellor Codes): {kappa_score_counsellor:.2f}")
    log.info(f"Manual Cohen's Kappa ({tier} Counsellor): {man_co:.2f}")
    log.info(f"Asymptotic Variance ({tier} Counsellor): {av_counsellor}")
    log.info(f"P-value ({tier} Counsellor): {p_co:.15e}")

    # (Optional) Plot confusion matrix and classification report for counsellor codes
    counsellor_conf_matrix = confusion_matrix(
        counsellor_human_labels, counsellor_automisc_labels, labels=counsellor_codes
    )
    counsellor_conf_matrix_df = pd.DataFrame(
        counsellor_conf_matrix,
        index=counsellor_codes,
        columns=counsellor_codes
    )

    total_correct_counsellor = np.trace(counsellor_conf_matrix)
    log.info(f"Total Correct Predictions (Counsellor): {total_correct_counsellor} / {len(counsellor_human_labels)} = {co_po*100:.2f}%")

    plt.figure(figsize=(8, 6))
    sns.heatmap(counsellor_conf_matrix_df, annot=True, fmt="d", cmap="Blues")
    # plt.title(f"Confusion Matrix: {tier} Counsellor Talk")
    plt.xlabel("AutoMISC", fontsize=20)
    plt.ylabel("Consensus Labels", fontsize=20)
    plt.tight_layout()
    plt.savefig(exp_output_dir / f"results/{tier}_counsellor_confusion_matrix.pdf", format="pdf", bbox_inches='tight')
    plt.close()

    class_report_counsellor = classification_report(counsellor_human_labels, counsellor_automisc_labels, labels=counsellor_codes, output_dict=True, zero_division=0)
    class_report_counsellor_df = pd.DataFrame(class_report_counsellor).transpose()
    log.info(f"\n{class_report_counsellor_df}")

    # --------------------- Client Calculations ---------------------
    client_human_labels = filtered_human_labels[filtered_human_labels.isin(client_codes)]
    client_automisc_labels = filtered_automisc_labels[filtered_automisc_labels.isin(client_codes)]

    cl_po, cl_pe, cm_cl, row_marginals_cl, col_marginals_cl = compute_p0_pe(client_human_labels, client_automisc_labels)
    man_cl = (cl_po - cl_pe) / (1 - cl_pe)

    kappa_score_client = cohen_kappa_score(client_human_labels, client_automisc_labels)
    av_client = compute_asymptotic_variance_kappa(cl_po, cl_pe, cm_cl, row_marginals_cl, col_marginals_cl, len(client_human_labels))
    p_client = compute_p_value(kappa_score_client, av_client)

    log.info(f"Cohen's Kappa ({tier} Client Codes): {kappa_score_client:.2f}")
    log.info(f"Manual Cohen's Kappa ({tier} Client): {man_cl:.2f}")
    log.info(f"Asymptotic Variance ({tier} Client): {av_client}")
    log.info(f"P-value ({tier} Client): {p_client:.15e}")

    # (Optional) Plot confusion matrix and classification report for client codes
    client_conf_matrix = confusion_matrix(
        client_human_labels, client_automisc_labels, labels=client_codes
    )
    client_conf_matrix_df = pd.DataFrame(
        client_conf_matrix,
        index=client_codes,
        columns=client_codes
    )
    total_correct_client = np.trace(client_conf_matrix)
    log.info(f"Total Correct Predictions (Client): {total_correct_client}")
    log.info(f"Total Correct Predictions (Client): {total_correct_client} / {len(client_human_labels)} = {cl_po*100:.2f}%")


    plt.figure(figsize=(8, 6))
    sns.heatmap(client_conf_matrix_df, annot=True, fmt="d", cmap="Oranges")
    # plt.title(f"Confusion Matrix: {tier} Client Talk")
    plt.xlabel("AutoMISC", fontsize=20)
    plt.ylabel("Consensus Labels", fontsize=20)
    plt.tight_layout()
    plt.savefig(exp_output_dir / f"results/{tier}_client_confusion_matrix.pdf", format="pdf", bbox_inches='tight')
    plt.close()

    class_report_client = classification_report(client_human_labels, client_automisc_labels, labels=client_codes, output_dict=True, zero_division=0)
    class_report_client_df = pd.DataFrame(class_report_client).transpose()
    log.info(f"\n{class_report_client_df}")

    class_report_counsellor_df.to_csv(exp_output_dir / f"results/{tier}_counsellor_cls.csv")
    class_report_client_df.to_csv(exp_output_dir / f"results/{tier}_client_cls.csv")
    counsellor_conf_matrix_df.to_csv(exp_output_dir / f"results/{tier}_counsellor_conf_mtx.csv")
    client_conf_matrix_df.to_csv(exp_output_dir / f"results/{tier}_client_conf_mtx.csv")
    return

def IRR(cfg: DictConfig) -> None:
    plt.rcParams["font.family"] = "Times New Roman"
    plt.rcParams["font.size"] = 20
    auto_anno_path = Path('data/annotated') / (
        f"{cfg.input_dataset.name}_"
        f"{cfg.input_dataset.subset}_"
        f"{cfg.annotated.class_structure}_"
        f"{cfg.annotated.model.rsplit('/', 1)[-1]}_"
        f"{cfg.annotated.context_mode}_"
        f"{cfg.annotated.num_context_turns if cfg.annotated.context_mode == 'interval' else ''}" 
        f"_annotated.csv"
        # f"_annotated_OLD.csv"
    )

    manual_path = Path('data/manual') / manual_fn
    log.info(f"AutoMISC annotations path: {auto_anno_path}")

    if cfg.annotated.class_structure == 'tiered':
        # rater_name = 'auto'
        cohens_kappa(cfg, auto_anno_path, manual_path, "T1", "GT") # GT for Ground Truth
        cohens_kappa(cfg, auto_anno_path, manual_path, "T2", "GT") # GT for Ground Truth
    elif cfg.annotated.class_structure == 'flat':
        log.info('gotta implement flat structure IRR')
    else:
        pass
    
    return
