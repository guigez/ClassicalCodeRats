"""
FIGURAS DO ARTIGO
=================

Gera as quatro figuras do manuscrito a partir de uma pasta de resultados
produzida por `run_all.py` (uma subpasta por tarefa, cada uma com
`_tabelas/` e `fold_XX/`).

  fig_partition   esquema da particao por sujeito (folds x camundongos)
  fig_seeds       F1 macro de cada replica de semente, por camundongo de teste
  fig_confusion   matrizes de confusao da tarefa de 4 classes, media dos folds
  fig_shap        mapas de atribuicao em quadros de um camundongo de teste

As tres primeiras so leem CSV e rodam em qualquer maquina com matplotlib e
pandas. A de SHAP recarrega os modelos (`model.pt`) e as imagens, entao roda
onde eles estao — no servidor.

Cada figura sai em PDF (vetorial, para o LaTeX) e PNG de 600 dpi.

Uso:
    python scripts/make_figures.py --results main_results --out figures
    python scripts/make_figures.py --results main_results --out figures \\
        --only shap --images_dir ../_dataset/images
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap, ListedColormap

# --------------------------------------------------------------------------
# Estilo. Larguras do IEEE: coluna 3.5 in, pagina 7.16 in. Cores categoricas
# validadas para daltonismo; a forma do marcador repete a identidade, para
# que a cor nunca seja a unica pista.
# --------------------------------------------------------------------------
COL_W, PAGE_W = 3.5, 7.16

ARMS = ["bernini_cnn", "resnet50_rf", "resnet50_rf_linear_head"]
ARM_LABEL = {"bernini_cnn": "Compact CNN",
             "resnet50_rf": "ResNet50 + random forest",
             "resnet50_rf_linear_head": "ResNet50 linear head"}
ARM_COLOR = {"bernini_cnn": "#eb6834",
             "resnet50_rf": "#1baf7a",
             "resnet50_rf_linear_head": "#2a78d6"}
ARM_MARKER = {"bernini_cnn": "o", "resnet50_rf": "s", "resnet50_rf_linear_head": "D"}

TASKS = ["binary", "three_mlss", "three_bernini", "multiclass"]
TASK_LABEL = {"binary": "Binary", "three_mlss": "Three-class MLSS",
              "three_bernini": "Three-class low", "multiclass": "Four-class"}
TASK_K = {"binary": 2, "three_mlss": 3, "three_bernini": 3, "multiclass": 4}

CLASSES_4 = ["Warm-up", "Moderate", "Heavy", "Severe"]
TEST_MICE = [1, 9, 10, 11]

INK, INK_MUTED, GRID = "#1f1f1f", "#6b6b6b", "#e4e4e2"
BLUE_RAMP = ["#ffffff", "#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5",
             "#256abf", "#184f95", "#0d366b"]

plt.rcParams.update({
    "font.family": "serif", "font.serif": ["Times New Roman", "Times", "STIXGeneral"],
    "mathtext.fontset": "stix",
    "font.size": 8, "axes.titlesize": 8, "axes.labelsize": 8,
    "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7,
    "axes.edgecolor": INK_MUTED, "axes.labelcolor": INK, "text.color": INK,
    "xtick.color": INK_MUTED, "ytick.color": INK_MUTED,
    "axes.spines.top": False, "axes.spines.right": False,
    "pdf.fonttype": 42, "ps.fonttype": 42,
})


def salvar(fig, out: Path, nome: str):
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / f"{nome}.pdf", bbox_inches="tight")
    fig.savefig(out / f"{nome}.png", dpi=600, bbox_inches="tight")
    plt.close(fig)
    print(f"[ok] {out / nome}.pdf / .png")


# ==========================================================================
# Figura 1 — particao
# ==========================================================================
def fig_partition(results: Path, out: Path, task: str = "multiclass"):
    """Grade folds x camundongos: teste, validacao ou treino."""
    resumo = pd.read_csv(results / task / "splits" / "splits_summary.csv")
    mice = list(range(1, 12))
    codigo = np.zeros((len(resumo), len(mice)), dtype=int)   # 0 treino
    for i, r in resumo.iterrows():
        for m in json.loads(r.test_subjects):
            codigo[i, m - 1] = 2
        for m in json.loads(r.val_subjects):
            codigo[i, m - 1] = 1

    cmap = ListedColormap(["#e4e4e2", "#eb6834", "#2a78d6"])
    fig, ax = plt.subplots(figsize=(COL_W, 1.45))
    ax.imshow(codigo, cmap=cmap, vmin=0, vmax=2, aspect="auto")
    # Grade branca de 2 px entre celulas e letra em cada celula colorida:
    # a identidade nao depende so da cor.
    ax.set_xticks(np.arange(-0.5, len(mice)), minor=True)
    ax.set_yticks(np.arange(-0.5, len(resumo)), minor=True)
    ax.grid(which="minor", color="white", linewidth=2)
    ax.tick_params(which="minor", length=0)
    for i in range(codigo.shape[0]):
        for j in range(codigo.shape[1]):
            if codigo[i, j]:
                ax.text(j, i, "Test" if codigo[i, j] == 2 else "Val",
                        ha="center", va="center", fontsize=5.5, color="white")
    ax.set_xticks(range(len(mice)), [str(m) for m in mice])
    ax.set_yticks(range(len(resumo)), [f"Fold {i + 1}" for i in range(len(resumo))])
    ax.set_xlabel("Mouse")
    for s in ax.spines.values():
        s.set_visible(False)
    ax.tick_params(length=0)
    ax.text(len(mice) - 0.5, -0.9, "Gray: training", ha="right", va="bottom",
            fontsize=6.5, color=INK_MUTED)
    salvar(fig, out, "fig_partition")


# ==========================================================================
# Figura 2 — replicas de semente
# ==========================================================================
def fig_seeds(results: Path, out: Path):
    """Um painel por tarefa; cinco pontos por modelo e camundongo, media em traco."""
    fig, axes = plt.subplots(2, 2, figsize=(PAGE_W, 4.0), sharey=True)
    deslocamento = {a: d for a, d in zip(ARMS, (-0.24, 0.0, 0.24))}
    rng = np.random.default_rng(0)

    for ax, task in zip(axes.ravel(), TASKS):
        df = pd.read_csv(results / task / "_tabelas" / "seed_replicates_by_subject.csv")
        for a in ARMS:
            sub = df[df.arm == a]
            for k, m in enumerate(TEST_MICE):
                v = sub[sub.subject_id == m].f1_macro.to_numpy()
                if not len(v):
                    continue
                x = k + deslocamento[a] + rng.uniform(-0.04, 0.04, len(v))
                ax.scatter(x, v, s=14, marker=ARM_MARKER[a], color=ARM_COLOR[a],
                           edgecolor="white", linewidth=0.6, zorder=3,
                           label=ARM_LABEL[a] if k == 0 else None)
                ax.hlines(v.mean(), k + deslocamento[a] - 0.09,
                          k + deslocamento[a] + 0.09, color=INK, linewidth=1.2, zorder=4)
        acaso = 1 / TASK_K[task]
        ax.axhline(acaso, color=INK_MUTED, linewidth=0.8, linestyle=(0, (3, 3)), zorder=1)

        ax.set_title(TASK_LABEL[task], loc="left", fontweight="bold")
        ax.set_xticks(range(len(TEST_MICE)), [f"Mouse {m}" for m in TEST_MICE])
        ax.set_xlim(-0.5, len(TEST_MICE) - 0.5)
        ax.set_ylim(0, 1)
        ax.yaxis.grid(True, color=GRID, linewidth=0.6)
        ax.set_axisbelow(True)
    for ax in axes[:, 0]:
        ax.set_ylabel("Macro-averaged F1")

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False,
               bbox_to_anchor=(0.5, 1.03), markerscale=1.3)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    salvar(fig, out, "fig_seeds")


# ==========================================================================
# Figura 3 — matrizes de confusao
# ==========================================================================
def _confusao(results: Path, task: str, arquivo: str) -> np.ndarray:
    """Media das matrizes normalizadas por linha de cada camundongo de teste.

    Somar contagens daria mais peso aos animais com mais quadros; a media por
    animal segue a unidade de inferencia do artigo e bate com os recalls do texto.
    """
    normalizadas = []
    for fold in sorted((results / task).glob("fold_*")):
        fam = "bernini_cnn" if arquivo == "cnn" else "resnet50_rf"
        nome = "confusion_test_linear_head.csv" if arquivo == "linear" else "confusion_test.csv"
        m = pd.read_csv(fold / fam / nome, index_col=0).to_numpy(dtype=float)
        normalizadas.append(m / m.sum(axis=1, keepdims=True))
    return np.mean(normalizadas, axis=0)


def fig_confusion(results: Path, out: Path, task: str = "multiclass"):
    """Normalizada por linha (recall), media sobre os quatro camundongos de teste."""
    fontes = [("cnn", "bernini_cnn"), ("rf", "resnet50_rf"), ("linear", "resnet50_rf_linear_head")]
    cmap = LinearSegmentedColormap.from_list("azul", BLUE_RAMP)
    fig, axes = plt.subplots(1, 3, figsize=(PAGE_W, 2.55))
    for ax, (arq, arm) in zip(axes, fontes):
        r = _confusao(results, task, arq)
        im = ax.imshow(r, cmap=cmap, vmin=0, vmax=1)
        for i in range(4):
            for j in range(4):
                ax.text(j, i, f"{r[i, j]:.2f}", ha="center", va="center", fontsize=7,
                        color="white" if r[i, j] > 0.55 else INK)
        ax.set_xticks(range(4), CLASSES_4, rotation=30, ha="right")
        ax.set_yticks(range(4), CLASSES_4 if ax is axes[0] else [""] * 4)
        ax.set_xlabel("Predicted class")
        ax.set_title(ARM_LABEL[arm], fontweight="bold")
        for s in ax.spines.values():
            s.set_visible(False)
        ax.tick_params(length=0)
    axes[0].set_ylabel("True class")
    cb = fig.colorbar(im, ax=axes, fraction=0.025, pad=0.02)
    cb.set_label("Proportion of true class")
    cb.outline.set_visible(False)
    salvar(fig, out, "fig_confusion")


# ==========================================================================
# Figura 4 — SHAP (precisa de modelos e imagens; roda no servidor)
# ==========================================================================
def _carregar(results: Path, task: str, fold: str, fam: str, n_classes: int, device):
    """Modelo vencedor do fold. O best_selection.json guarda caminho absoluto
    do servidor; usamos so o nome da pasta da rede, relativo a --results."""
    from models import create_model
    import torch

    sel = json.loads((results / task / fold / fam / "best_selection.json").read_text())
    net_dir = results / task / fold / fam / Path(sel["net_dir"]).name
    config = json.loads((net_dir / "config_used.json").read_text())
    model = create_model(config, n_classes).to(device)
    model.load_state_dict(torch.load(net_dir / "model.pt",
                                     map_location=device)["model_state_dict"])
    model.eval()
    for p in model.parameters():
        p.requires_grad_(True)
    return model, config


def fig_shap(results: Path, out: Path, images_dir: str, fold: str = "fold_02",
             task: str = "multiclass", background_size: int = 32, seed: int = 42,
             device_name: str = "auto"):
    """Um quadro por classe do camundongo de teste do fold, acertado pela cabeca
    linear. Colunas: imagem, mascara do animal, |SHAP| da CNN e da ResNet50
    para a classe real, com o enriquecimento E de cada mapa."""
    src = Path(__file__).resolve().parent.parent / "src"
    sys.path.insert(0, str(src))
    import torch
    from common import resolve_device
    from explain_shap import (animal_mask, attribution_enrichment, explain_pixels,
                              sample_background, to_display)
    from train import build_loaders

    device = resolve_device(device_name)
    n_classes = 4
    fold_csv = results / task / "splits" / f"{fold}.csv"

    # Quadros: sorteio entre os acertos da cabeca linear em cada classe.
    preds = pd.read_csv(results / task / fold / "resnet50_rf" / "predictions_test_linear_head.csv")
    rng = np.random.default_rng(seed)
    escolhidos = []
    for c in range(n_classes):
        sub = preds[(preds.y_true == c) & (preds.correct == 1)]
        if len(sub):
            escolhidos.append(sub.iloc[int(rng.integers(len(sub)))])
    nomes = [r.filename for r in escolhidos]
    rotulos = [int(r.y_true) for r in escolhidos]
    mouse = int(escolhidos[0].subject_id)

    mapas, imagens = {}, None
    for fam in ("bernini_cnn", "resnet50_rf"):
        model, config = _carregar(results, task, fold, fam, n_classes, device)
        ds, _ = build_loaders(str(fold_csv), images_dir, config, n_classes)
        por_nome = {n: i for i, n in enumerate(ds["test"].df.filename)}
        X = torch.stack([ds["test"][por_nome[n]][0] for n in nomes])
        bg = sample_background(ds["train_eval"], background_size, seed, device)
        sv = explain_pixels(model, bg, X, device, n_classes)
        norm = config.get("model_name") == "resnet50_rf"
        if imagens is None:
            imagens = [to_display(x, norm) for x in X]
        mapas[fam] = [np.abs(sv[i, ..., rotulos[i]]).sum(0) for i in range(len(nomes))]

    fig, axes = plt.subplots(len(nomes), 4, figsize=(PAGE_W, 1.85 * len(nomes)))
    titulos = ["Input frame", "Animal mask (Otsu)", "Compact CNN |SHAP|", "ResNet50 |SHAP|"]
    for i, nome in enumerate(nomes):
        img = imagens[i]
        mask = animal_mask(img, background_removed=False)
        cinza = img.mean(axis=2)
        axes[i, 0].imshow(img)
        axes[i, 0].set_ylabel(CLASSES_4[rotulos[i]], fontweight="bold")
        axes[i, 1].imshow(cinza, cmap="gray")
        axes[i, 1].contour(mask, levels=[0.5], colors="#eb6834", linewidths=0.8)
        for j, fam in ((2, "bernini_cnn"), (3, "resnet50_rf")):
            h = mapas[fam][i]
            e = attribution_enrichment(h, mask)["enrichment"]
            axes[i, j].imshow(cinza, cmap="gray")
            axes[i, j].imshow(h / (np.percentile(h, 99.5) + 1e-12), cmap="magma",
                              vmin=0, vmax=1, alpha=0.75)
            axes[i, j].text(0.03, 0.04, f"E = {e:.2f}", transform=axes[i, j].transAxes,
                            fontsize=7, color="white",
                            bbox=dict(facecolor="black", alpha=0.55, pad=1.5, linewidth=0))
        for j in range(4):
            axes[i, j].set_xticks([]); axes[i, j].set_yticks([])
            for s in axes[i, j].spines.values():
                s.set_visible(False)
            if i == 0:
                axes[i, j].set_title(titulos[j], fontweight="bold")
    fig.suptitle(f"Test mouse {mouse}", x=0.01, ha="left", fontsize=8, color=INK_MUTED)
    fig.tight_layout()
    salvar(fig, out, "fig_shap")
    pd.DataFrame({"filename": nomes, "y_true": rotulos}).to_csv(out / "fig_shap_frames.csv", index=False)


def main() -> int:
    ap = argparse.ArgumentParser(description="Figuras do artigo.")
    ap.add_argument("--results", required=True, help="Pasta com uma subpasta por tarefa.")
    ap.add_argument("--out", default="figures")
    ap.add_argument("--only", nargs="+", choices=["partition", "seeds", "confusion", "shap"],
                    default=["partition", "seeds", "confusion"],
                    help="Por padrao gera as tres figuras de tabela; 'shap' exige modelos e imagens.")
    ap.add_argument("--images_dir", default=None, help="Necessario para --only shap.")
    ap.add_argument("--shap_fold", default="fold_02")
    ap.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    args = ap.parse_args()

    results, out = Path(args.results), Path(args.out)
    if "partition" in args.only:
        fig_partition(results, out)
    if "seeds" in args.only:
        fig_seeds(results, out)
    if "confusion" in args.only:
        fig_confusion(results, out)
    if "shap" in args.only:
        if not args.images_dir:
            ap.error("--only shap precisa de --images_dir")
        fig_shap(results, out, args.images_dir, fold=args.shap_fold, device_name=args.device)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
