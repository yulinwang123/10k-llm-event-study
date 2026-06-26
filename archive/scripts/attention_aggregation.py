"""
attention_aggregation.py
────────────────────────
Hierarchical Attention Aggregation for 10-K MD&A → CAR prediction.

Motivation
──────────
10-K MD&A sections (~10,000 words median) contain semantically heterogeneous
content: forward-looking guidance, risk disclosures, historical performance
narratives, and boilerplate language. Simple aggregation strategies (bag-of-words
mean, first-N-chars truncation) treat all paragraphs equally, ignoring the fact
that different sections carry very different predictive signal for market reactions.

Optimization Goal
─────────────────
Learn a soft attention distribution α over paragraph embeddings {h_1, ..., h_T}
such that the weighted aggregate z = Σ α_t · h_t minimizes prediction error on
CAR[-1,+1] around the 10-K filing date.

Formally:
    min_{W_a, W_pred} E[ (CAR_i - f(z_i; W_pred))² ]
    where α_t = softmax( tanh(W_a · h_t + b_a) · u )
          z_i = Σ_t α_t · h_t

This is a soft-attention variant of Yang et al. (2016) HAN, adapted for
financial text → abnormal return prediction.

The attention weights α_t reveal WHICH parts of the MD&A drive the prediction,
directly answering: "which sections of 10-Ks best predict earnings reactions?"

Architecture
────────────
  Paragraph text
       ↓
  SBERT encoder (all-mpnet-base-v2, frozen)  →  h_t ∈ R^768
       ↓
  Attention MLP: score_t = u^T · tanh(W_a · h_t + b_a)
       ↓
  α = softmax(scores)
       ↓
  z = Σ α_t · h_t   (context vector, 768-dim)
       ↓
  Prediction head: CAR_hat = W_pred · [z || controls] + b
       ↓
  Loss: MSE(CAR_hat, CAR_filed_1_1)

Reference
─────────
Yang et al. (2016). Hierarchical Attention Networks for Document Classification.
NAACL 2016.  (adapted from classification → regression, sentences → paragraphs)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sentence_transformers import SentenceTransformer
import numpy as np
import pandas as pd
from pathlib import Path
import re


# ── Paragraph splitter ────────────────────────────────────────────────────────

def split_paragraphs(text: str, min_words: int = 20, max_paragraphs: int = 64) -> list[str]:
    """
    Split MD&A text into meaningful paragraphs.
    Filters out boilerplate (page numbers, headers < 20 words).
    Caps at max_paragraphs to keep memory bounded.
    """
    # Split on double newlines or section breaks
    raw = re.split(r'\n{2,}|\r\n{2,}', text.strip())
    paras = []
    for p in raw:
        p = p.strip()
        words = p.split()
        if len(words) >= min_words:
            paras.append(p)
    # If no paragraph breaks found, fall back to sentence chunking
    if len(paras) < 3:
        sentences = re.split(r'(?<=[.!?])\s+', text)
        # Group into chunks of ~5 sentences
        paras = [' '.join(sentences[i:i+5]) for i in range(0, len(sentences), 5)]
        paras = [p for p in paras if len(p.split()) >= min_words]
    return paras[:max_paragraphs]


# ── Attention aggregation model ───────────────────────────────────────────────

class AttentionAggregator(nn.Module):
    """
    Soft attention over paragraph embeddings.

    Input:  H  ∈ R^{T × d}   (T paragraphs, each d-dimensional)
    Output: z  ∈ R^d          (attention-weighted aggregate)
            α  ∈ R^T          (attention weights, sums to 1)
    """
    def __init__(self, embed_dim: int = 768, attention_dim: int = 256):
        super().__init__()
        self.W_a = nn.Linear(embed_dim, attention_dim, bias=True)
        self.u   = nn.Linear(attention_dim, 1, bias=False)

    def forward(self, H: torch.Tensor, mask: torch.Tensor = None):
        """
        H:    (batch, T, d)
        mask: (batch, T) — True for valid paragraphs, False for padding
        """
        # score_t = u^T · tanh(W_a · h_t)
        scores = self.u(torch.tanh(self.W_a(H))).squeeze(-1)   # (batch, T)

        if mask is not None:
            scores = scores.masked_fill(~mask, float('-inf'))

        alpha = F.softmax(scores, dim=-1)                        # (batch, T)

        # Context vector: weighted sum
        z = torch.bmm(alpha.unsqueeze(1), H).squeeze(1)         # (batch, d)

        return z, alpha


class MDAPredictor(nn.Module):
    """
    Full model: SBERT paragraph embeddings → attention aggregation → CAR prediction.

    Controls (log_assets, log_mktcap, bm_ratio, roa, leverage) are concatenated
    with the context vector before the prediction head to partial out known
    cross-sectional predictors of returns.
    """
    def __init__(self, embed_dim: int = 768, attention_dim: int = 256,
                 n_controls: int = 5, hidden_dim: int = 128):
        super().__init__()
        self.attention = AttentionAggregator(embed_dim, attention_dim)

        # Prediction head
        self.head = nn.Sequential(
            nn.Linear(embed_dim + n_controls, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, H, controls, mask=None):
        """
        H:        (batch, T, 768)  paragraph embeddings
        controls: (batch, 5)       standardized financial controls
        mask:     (batch, T)       valid paragraph mask
        Returns:  car_hat (batch,), alpha (batch, T)
        """
        z, alpha = self.attention(H, mask)             # (batch, 768), (batch, T)
        x = torch.cat([z, controls], dim=-1)           # (batch, 773)
        car_hat = self.head(x).squeeze(-1)             # (batch,)
        return car_hat, alpha


# ── Dataset ───────────────────────────────────────────────────────────────────

class MDADataset(Dataset):
    """
    Loads pre-computed paragraph embeddings (or computes them on the fly).

    Expected parquet schema (embeddings_path):
      gvkey, fyear, para_idx, embedding (768-d float array)

    Panel must have: gvkey, fyear, car_filed_1_1, log_assets, log_mktcap,
                     bm_ratio, roa, leverage
    """
    CONTROLS = ['log_assets', 'log_mktcap', 'bm_ratio', 'roa', 'leverage']

    def __init__(self, panel: pd.DataFrame, embeddings: dict,
                 max_paragraphs: int = 64):
        self.panel = panel.reset_index(drop=True)
        self.embeddings = embeddings   # {(gvkey, fyear): np.array (T, 768)}
        self.max_T = max_paragraphs

        # Z-score controls
        for col in self.CONTROLS:
            mu, sigma = panel[col].mean(), panel[col].std()
            self.panel[f'{col}_z'] = (panel[col] - mu) / (sigma + 1e-8)

    def __len__(self):
        return len(self.panel)

    def __getitem__(self, idx):
        row = self.panel.iloc[idx]
        key = (row['gvkey'], int(row['fyear']))

        # Paragraph embeddings
        emb = self.embeddings.get(key)
        if emb is None:
            emb = np.zeros((1, 768), dtype=np.float32)

        T = min(len(emb), self.max_T)
        H = np.zeros((self.max_T, 768), dtype=np.float32)
        H[:T] = emb[:T]
        mask = np.zeros(self.max_T, dtype=bool)
        mask[:T] = True

        controls = np.array([row[f'{c}_z'] for c in self.CONTROLS], dtype=np.float32)
        car = np.float32(row['car_filed_1_1'])

        return (torch.tensor(H),
                torch.tensor(mask),
                torch.tensor(controls),
                torch.tensor(car))


# ── Training loop ─────────────────────────────────────────────────────────────

def train(model, loader, optimizer, device):
    model.train()
    total_loss = 0.0
    for H, mask, controls, car in loader:
        H, mask, controls, car = (x.to(device) for x in (H, mask, controls, car))
        optimizer.zero_grad()
        car_hat, _ = model(H, controls, mask)
        loss = F.mse_loss(car_hat, car)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * len(car)
    return total_loss / len(loader.dataset)


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    preds, targets = [], []
    for H, mask, controls, car in loader:
        H, mask, controls, car = (x.to(device) for x in (H, mask, controls, car))
        car_hat, _ = model(H, controls, mask)
        preds.append(car_hat.cpu())
        targets.append(car.cpu())
    preds    = torch.cat(preds).numpy()
    targets  = torch.cat(targets).numpy()
    mse      = float(np.mean((preds - targets) ** 2))
    # R² vs. mean baseline
    ss_res   = np.sum((preds - targets) ** 2)
    ss_tot   = np.sum((targets - targets.mean()) ** 2)
    r2       = 1 - ss_res / ss_tot
    return {'mse': mse, 'r2': r2}


# ── Attention interpretation ───────────────────────────────────────────────────

@torch.no_grad()
def get_attention_weights(model, H, mask, controls, device):
    """
    Return attention weights over paragraphs for a single filing.
    Use this to visualize which paragraphs the model focuses on.
    """
    model.eval()
    H        = H.unsqueeze(0).to(device)
    mask     = mask.unsqueeze(0).to(device)
    controls = controls.unsqueeze(0).to(device)
    _, alpha = model(H, controls, mask)
    return alpha.squeeze(0).cpu().numpy()   # shape: (max_T,)


def top_paragraphs(paragraphs: list[str], alpha: np.ndarray,
                   top_k: int = 5) -> pd.DataFrame:
    """
    Return the top-k paragraphs by attention weight with their scores.
    Directly answers: 'which parts of this 10-K drive the CAR prediction?'
    """
    T = len(paragraphs)
    weights = alpha[:T]
    idx = np.argsort(weights)[::-1][:top_k]
    return pd.DataFrame({
        'rank':      range(1, top_k + 1),
        'para_idx':  idx,
        'attention': weights[idx].round(4),
        'text':      [paragraphs[i][:200] + '...' for i in idx]
    })


# ── Paragraph embedding precomputation ───────────────────────────────────────

def precompute_paragraph_embeddings(mda_dir: str, output_path: str,
                                    batch_size: int = 64,
                                    max_paragraphs: int = 64):
    """
    Encode all paragraphs in MD&A text files using SBERT.
    Run once on Midway3 GPU; save as dict {(gvkey, fyear): np.array (T, 768)}.

    Much richer than the single-vector embed_novelty in Track 4:
    here every paragraph gets its own embedding, enabling attention over them.
    """
    import pickle

    encoder = SentenceTransformer('all-mpnet-base-v2', device='cuda')
    mda_dir = Path(mda_dir)
    embeddings = {}

    txt_files = list(mda_dir.rglob('*.txt'))
    print(f'Encoding {len(txt_files)} MD&A files ...')

    for fp in txt_files:
        # Expected path: filings/{ticker}/{fyear}/*.txt
        parts = fp.parts
        try:
            ticker = parts[-3]
            fyear  = int(parts[-2])
        except (IndexError, ValueError):
            continue

        text  = fp.read_text(errors='ignore')
        paras = split_paragraphs(text, max_paragraphs=max_paragraphs)
        if not paras:
            continue

        vecs = encoder.encode(paras, batch_size=batch_size,
                               show_progress_bar=False,
                               convert_to_numpy=True)
        embeddings[(ticker, fyear)] = vecs.astype(np.float32)

    with open(output_path, 'wb') as f:
        pickle.dump(embeddings, f)
    print(f'Saved embeddings for {len(embeddings)} filings → {output_path}')
    return embeddings


# ── Main: train and interpret ─────────────────────────────────────────────────

if __name__ == '__main__':
    import pickle
    from sklearn.model_selection import train_test_split

    DATA = 'data'
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Using device: {DEVICE}')

    # Load panel
    panel = pd.read_parquet(f'{DATA}/analysis_panel_1234.parquet')
    panel = panel.dropna(subset=['car_filed_1_1'] +
                                ['log_assets','log_mktcap','bm_ratio','roa','leverage'])
    print(f'Panel: {len(panel):,} firm-years')

    # Load pre-computed paragraph embeddings
    with open(f'{DATA}/paragraph_embeddings.pkl', 'rb') as f:
        embeddings = pickle.load(f)
    print(f'Embeddings: {len(embeddings):,} filings')

    # Train / val split (by firm to avoid look-ahead)
    firms = panel['gvkey'].unique()
    train_firms, val_firms = train_test_split(firms, test_size=0.2, random_state=42)
    train_panel = panel[panel['gvkey'].isin(train_firms)]
    val_panel   = panel[panel['gvkey'].isin(val_firms)]

    train_ds = MDADataset(train_panel, embeddings)
    val_ds   = MDADataset(val_panel,   embeddings)
    train_dl = DataLoader(train_ds, batch_size=32, shuffle=True,  num_workers=2)
    val_dl   = DataLoader(val_ds,   batch_size=64, shuffle=False, num_workers=2)

    # Model
    model = MDAPredictor(embed_dim=768, attention_dim=256,
                         n_controls=5, hidden_dim=128).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=30)

    # Training
    print('\nEpoch  Train MSE  Val MSE   Val R²')
    print('─' * 40)
    best_r2, best_state = -np.inf, None
    for epoch in range(1, 31):
        train_loss = train(model, train_dl, optimizer, DEVICE)
        val_metrics = evaluate(model, val_dl, DEVICE)
        scheduler.step()

        if val_metrics['r2'] > best_r2:
            best_r2    = val_metrics['r2']
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

        if epoch % 5 == 0:
            print(f'  {epoch:3d}    {train_loss:.5f}    '
                  f'{val_metrics["mse"]:.5f}   {val_metrics["r2"]:+.4f}')

    model.load_state_dict(best_state)
    print(f'\nBest val R² = {best_r2:.4f}')
    torch.save(best_state, f'{DATA}/attention_model.pt')

    # ── Interpret: which paragraphs matter? ───────────────────────────────────
    # Aggregate attention weights across all filings to find which paragraph
    # POSITIONS (intro vs mid vs end) receive highest attention on average.
    model.eval()
    all_alpha = []
    for H, mask, controls, _ in val_dl:
        H, mask, controls = H.to(DEVICE), mask.to(DEVICE), controls.to(DEVICE)
        _, alpha = model(H, controls, mask)
        # Mask out padding before recording
        alpha_np = alpha.cpu().numpy()
        mask_np  = mask.cpu().numpy()
        for a, m in zip(alpha_np, mask_np):
            T = m.sum()
            if T > 1:
                # Normalize position to [0, 1] for comparability across filings
                positions = np.linspace(0, 1, T)
                all_alpha.append(list(zip(positions, a[:T])))

    # Average attention by position decile
    from collections import defaultdict
    decile_weights = defaultdict(list)
    for filing in all_alpha:
        for pos, w in filing:
            decile = int(pos * 10)
            decile_weights[decile].append(w)

    print('\nAverage attention weight by paragraph position (0=start, 9=end):')
    print('Decile  Mean α')
    for d in range(10):
        ws = decile_weights[d]
        print(f'  {d*10:3d}%   {np.mean(ws):.4f}')
