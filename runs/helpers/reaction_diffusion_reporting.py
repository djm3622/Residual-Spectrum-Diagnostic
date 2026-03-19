"""Reporting/visualization helpers for reaction-diffusion runner."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Mapping

import numpy as np

from data.reaction_diffusion import GrayScottConfig
from runs.helpers.reaction_diffusion_training import (
    noisy_reference_trajectory_coupled as _noisy_reference_trajectory_coupled,
)


def save_standard_figures(
    results: Mapping[str, Any],
    viz_payload: Mapping[str, Any],
    run_out_dir: Path,
    method: str,
    requested_loss: str,
    requested_basis: str,
    seed: int,
) -> None:
    from utils.plotting import (
        save_band_profile_plot,
        save_clean_noisy_metric_bar,
        save_clean_noisy_summary_plot,
        save_dual_band_gap_bootstrap_plot,
        save_metric_vs_l2_grid,
        save_spectral_band_error_plot,
    )

    save_clean_noisy_summary_plot(
        results,
        title=(
            f"Reaction-Diffusion | method={method} | loss={requested_loss} "
            f"| basis={requested_basis} | seed={seed}"
        ),
        output_path=run_out_dir / "summary.png",
    )

    save_clean_noisy_metric_bar(
        results["clean_pde_residual_st_rms"],
        results["noisy_pde_residual_st_rms"],
        metric_label="Space-time PDE residual RMS",
        output_path=run_out_dir / "pde_residual_space_time.png",
        title="PDE residual norm (space-time)",
    )
    save_clean_noisy_metric_bar(
        results["clean_boundary_error"],
        results["noisy_boundary_error"],
        metric_label="Boundary-condition error (periodic)",
        output_path=run_out_dir / "boundary_condition_error.png",
        title="Boundary-condition error",
    )
    save_clean_noisy_metric_bar(
        results["clean_spectral_multiband_error"],
        results["noisy_spectral_multiband_error"],
        metric_label="Multi-band spectral error",
        output_path=run_out_dir / "spectral_multiband_error.png",
        title="Multi-band spectral error",
    )
    save_clean_noisy_metric_bar(
        results["clean_fourier_coeff_mse_multiband_vs_clean_gt"],
        results["noisy_fourier_coeff_mse_multiband_vs_clean_gt"],
        metric_label="Multi-band Fourier coeff-MSE",
        output_path=run_out_dir / "fourier_coeff_mse_multiband.png",
        title="Multi-band Fourier coeff-MSE (vs clean GT)",
    )

    diagnostics_viz = viz_payload.get("diagnostics", {})
    save_spectral_band_error_plot(
        clean_band_error=diagnostics_viz.get("clean_spectral_band_error_mean", []),
        noisy_band_error=diagnostics_viz.get("noisy_spectral_band_error_mean", []),
        band_labels=diagnostics_viz.get("spectral_band_labels", []),
        band_centers=diagnostics_viz.get("spectral_band_centers", []),
        output_path=run_out_dir / "spectral_band_error_profile.png",
        title="Per-band spectral error profile",
    )
    save_band_profile_plot(
        clean_band_values=diagnostics_viz.get("clean_fourier_coeff_mse_band_vs_clean_gt_mean", []),
        noisy_band_values=diagnostics_viz.get("noisy_fourier_coeff_mse_band_vs_clean_gt_mean", []),
        band_labels=diagnostics_viz.get("spectral_band_labels", []),
        band_centers=diagnostics_viz.get("spectral_band_centers", []),
        output_path=run_out_dir / "fourier_coeff_mse_band_profile.png",
        title="Per-band Fourier coeff-MSE vs clean GT",
        y_label="Fourier coefficient MSE",
    )
    save_dual_band_gap_bootstrap_plot(
        band_labels=diagnostics_viz.get("spectral_band_labels", []),
        band_centers=diagnostics_viz.get("spectral_band_centers", []),
        fraction_gap_bootstrap=diagnostics_viz.get("fraction_band_gap_bootstrap_noisy_minus_clean", []),
        coeff_mse_gap_bootstrap=diagnostics_viz.get("coeff_mse_band_gap_bootstrap_noisy_minus_clean", []),
        output_path=run_out_dir / "spectral_gap_bootstrap_ci.png",
        title="Bootstrap CI: spectral gap (noisy-clean, vs clean GT)",
    )

    diagnostic_series = diagnostics_viz.get("series", {})
    save_metric_vs_l2_grid(
        l2_clean=diagnostic_series.get("clean_l2", []),
        l2_noisy=diagnostic_series.get("noisy_l2", []),
        metric_series={
            "HFV": {
                "clean": diagnostic_series.get("clean_hfv", []),
                "noisy": diagnostic_series.get("noisy_hfv", []),
            },
            "LFV": {
                "clean": diagnostic_series.get("clean_lfv", []),
                "noisy": diagnostic_series.get("noisy_lfv", []),
            },
            "PDE Residual RMS": {
                "clean": diagnostic_series.get("clean_pde_residual_st_rms", []),
                "noisy": diagnostic_series.get("noisy_pde_residual_st_rms", []),
            },
            "Boundary Error": {
                "clean": diagnostic_series.get("clean_boundary_error", []),
                "noisy": diagnostic_series.get("noisy_boundary_error", []),
            },
            "Spectral Multi-band Error": {
                "clean": diagnostic_series.get("clean_spectral_multiband_error", []),
                "noisy": diagnostic_series.get("noisy_spectral_multiband_error", []),
            },
            "Fourier Coeff MSE": {
                "clean": diagnostic_series.get("clean_fourier_coeff_mse_multiband_vs_clean_gt", []),
                "noisy": diagnostic_series.get("noisy_fourier_coeff_mse_multiband_vs_clean_gt", []),
            },
        },
        output_path=run_out_dir / "metrics_vs_l2.png",
        title="Metric effectiveness vs L2",
    )


def save_fit_visualizations(
    artifacts: Mapping[str, Any],
    viz_payload: Mapping[str, Any],
    run_out_dir: Path,
) -> None:
    from utils.plotting import save_coupled_fit_panel

    fit_dir = run_out_dir / "fit_quality"
    fit_viz = artifacts.get("fit_visualization", {})
    input_viz = fit_viz.get("input", {}) if isinstance(fit_viz, dict) else {}
    output_viz = fit_viz.get("output", {}) if isinstance(fit_viz, dict) else {}
    input_cmap = str(input_viz.get("cmap", "cividis"))
    input_border_color = str(input_viz.get("border_color", "#2A9D8F"))
    input_border_width = float(input_viz.get("border_width", 2.0))
    output_cmap = str(output_viz.get("cmap", "viridis"))

    save_coupled_fit_panel(
        viz_payload["eval"]["input_u"],
        viz_payload["eval"]["input_v"],
        viz_payload["eval"]["target_u"],
        viz_payload["eval"]["target_v"],
        viz_payload["eval"]["pred_u_clean"],
        viz_payload["eval"]["pred_v_clean"],
        output_path=fit_dir / "eval_clean.png",
        title="Eval split | Clean model",
        output_cmap=output_cmap,
        input_cmap=input_cmap,
        input_border_color=input_border_color,
        input_border_width=input_border_width,
        target_u_noisy=viz_payload["eval"]["target_u_noisy"],
        target_v_noisy=viz_payload["eval"]["target_v_noisy"],
        model_label="Clean model",
    )
    save_coupled_fit_panel(
        viz_payload["eval"]["input_u"],
        viz_payload["eval"]["input_v"],
        viz_payload["eval"]["target_u"],
        viz_payload["eval"]["target_v"],
        viz_payload["eval"]["pred_u_noisy"],
        viz_payload["eval"]["pred_v_noisy"],
        output_path=fit_dir / "eval_noisy.png",
        title="Eval split | Noisy model",
        output_cmap=output_cmap,
        input_cmap=input_cmap,
        input_border_color=input_border_color,
        input_border_width=input_border_width,
        target_u_noisy=viz_payload["eval"]["target_u_noisy"],
        target_v_noisy=viz_payload["eval"]["target_v_noisy"],
        model_label="Noisy model",
    )
    save_coupled_fit_panel(
        viz_payload["test"]["input_u"],
        viz_payload["test"]["input_v"],
        viz_payload["test"]["target_u"],
        viz_payload["test"]["target_v"],
        viz_payload["test"]["pred_u_clean"],
        viz_payload["test"]["pred_v_clean"],
        output_path=fit_dir / "test_clean.png",
        title="Test split | Clean model",
        output_cmap=output_cmap,
        input_cmap=input_cmap,
        input_border_color=input_border_color,
        input_border_width=input_border_width,
        target_u_noisy=viz_payload["test"]["target_u_noisy"],
        target_v_noisy=viz_payload["test"]["target_v_noisy"],
        model_label="Clean model",
    )
    save_coupled_fit_panel(
        viz_payload["test"]["input_u"],
        viz_payload["test"]["input_v"],
        viz_payload["test"]["target_u"],
        viz_payload["test"]["target_v"],
        viz_payload["test"]["pred_u_noisy"],
        viz_payload["test"]["pred_v_noisy"],
        output_path=fit_dir / "test_noisy.png",
        title="Test split | Noisy model",
        output_cmap=output_cmap,
        input_cmap=input_cmap,
        input_border_color=input_border_color,
        input_border_width=input_border_width,
        target_u_noisy=viz_payload["test"]["target_u_noisy"],
        target_v_noisy=viz_payload["test"]["target_v_noisy"],
        model_label="Noisy model",
    )


def save_trajectory_visualizations(
    artifacts: Mapping[str, Any],
    viz_payload: Mapping[str, Any],
    run_out_dir: Path,
    config: GrayScottConfig,
    seed: int,
) -> None:
    from utils.plotting import save_trajectory_error_rows, save_trajectory_field_rows

    fit_dir = run_out_dir / "fit_quality"
    trajectory_payload = viz_payload.get("trajectory", {})
    rows = trajectory_payload.get("rows", [])
    case_indices = trajectory_payload.get("case_indices", [])
    step_indices = trajectory_payload.get("step_indices", [])

    by_case: Dict[int, Dict[str, np.ndarray]] = {}
    for row in rows:
        case_idx = int(row["case_index"])
        case_bucket = by_case.setdefault(case_idx, {})
        case_bucket["u_true"] = row["u_true"]
        case_bucket["v_true"] = row["v_true"]
        case_bucket[f"u_{row['model']}"] = row["u_pred"]
        case_bucket[f"v_{row['model']}"] = row["v_pred"]

    ordered_cases = [int(idx) for idx in case_indices if int(idx) in by_case]
    if not ordered_cases:
        ordered_cases = sorted(by_case.keys())

    u_field_rows = []
    v_field_rows = []
    u_rows = []
    v_rows = []
    for case_idx in ordered_cases:
        bucket = by_case[case_idx]
        u_true = bucket.get("u_true")
        v_true = bucket.get("v_true")
        u_clean = bucket.get("u_clean")
        v_clean = bucket.get("v_clean")
        u_noisy = bucket.get("u_noisy")
        v_noisy = bucket.get("v_noisy")

        if u_true is None or v_true is None or u_clean is None or v_clean is None or u_noisy is None or v_noisy is None:
            continue

        u_truth_noisy, v_truth_noisy = _noisy_reference_trajectory_coupled(
            u_true,
            v_true,
            config,
            rng_seed=seed * 1_000_000 + case_idx * 10_000 + 421,
        )

        u_field_rows.append({"label": f"case {case_idx} | Clean GT", "traj": u_true})
        u_field_rows.append({"label": f"case {case_idx} | Noisy GT", "traj": u_truth_noisy})
        u_field_rows.append({"label": f"case {case_idx} | Clean Pred", "traj": u_clean})
        u_field_rows.append({"label": f"case {case_idx} | Noisy Pred", "traj": u_noisy})

        v_field_rows.append({"label": f"case {case_idx} | Clean GT", "traj": v_true})
        v_field_rows.append({"label": f"case {case_idx} | Noisy GT", "traj": v_truth_noisy})
        v_field_rows.append({"label": f"case {case_idx} | Clean Pred", "traj": v_clean})
        v_field_rows.append({"label": f"case {case_idx} | Noisy Pred", "traj": v_noisy})

        u_rows.append({"label": f"case {case_idx} | Clean Pred vs Clean GT", "pred": u_clean, "target": u_true})
        u_rows.append({"label": f"case {case_idx} | Noisy Pred vs Clean GT", "pred": u_noisy, "target": u_true})
        u_rows.append({"label": f"case {case_idx} | Clean Pred vs Noisy GT", "pred": u_clean, "target": u_truth_noisy})
        u_rows.append({"label": f"case {case_idx} | Noisy Pred vs Noisy GT", "pred": u_noisy, "target": u_truth_noisy})
        v_rows.append({"label": f"case {case_idx} | Clean Pred vs Clean GT", "pred": v_clean, "target": v_true})
        v_rows.append({"label": f"case {case_idx} | Noisy Pred vs Clean GT", "pred": v_noisy, "target": v_true})
        v_rows.append({"label": f"case {case_idx} | Clean Pred vs Noisy GT", "pred": v_clean, "target": v_truth_noisy})
        v_rows.append({"label": f"case {case_idx} | Noisy Pred vs Noisy GT", "pred": v_noisy, "target": v_truth_noisy})

    save_trajectory_field_rows(
        u_field_rows,
        step_indices=step_indices,
        output_path=fit_dir / "trajectory_u_fields.png",
        title="Trajectory snapshots | Species u",
        cmap="viridis",
    )
    save_trajectory_field_rows(
        v_field_rows,
        step_indices=step_indices,
        output_path=fit_dir / "trajectory_v_fields.png",
        title="Trajectory snapshots | Species v",
        cmap="viridis",
    )

    save_trajectory_error_rows(
        u_rows,
        step_indices=step_indices,
        output_path=fit_dir / "trajectory_u_error.png",
        title="Trajectory absolute error snapshots | Species u",
        cmap="magma",
    )
    save_trajectory_error_rows(
        v_rows,
        step_indices=step_indices,
        output_path=fit_dir / "trajectory_v_error.png",
        title="Trajectory absolute error snapshots | Species v",
        cmap="magma",
    )
