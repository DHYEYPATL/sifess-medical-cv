# SIFESS Method

**Soft Isophote-Frame Equivariant Self-Supervision** — a training law for medical
SSL: representations should be *softly* equivariant to each image’s own
structure-tensor / isophote eigenframe, gated by coherence \(C\).

This is **not** an architecture zoo, freeze schedule, or ensemble trick.

## 1. Structure tensor / isophote frame

For a (grayscale) image \(I\), form the structure tensor at integration scale
\(\sigma_\rho\) (default **1.5**):

\[
J_\rho = G_{\rho} * \bigl(\nabla I\,\nabla I^{\mathsf T}\bigr)
\]

with eigenvalues \(\lambda_1 \ge \lambda_2 \ge 0\) and orthonormal eigenvectors
\(e_1\) (major / gradient-dominated) and \(e_2\) (minor / isophote tangent).

**Coherence**

\[
C = \frac{\lambda_1 - \lambda_2}{\lambda_1 + \lambda_2 + \varepsilon}
\in [0,1]
\]

**Gate (stop-grad)**

\[
\bar C = \operatorname{stopgrad}\bigl(\operatorname{mean}(C)\bigr)
\]

High \(\bar C\): anisotropic structure → equivariance constraint is meaningful.
Low \(\bar C\): isotropic / noisy → down-weight the residual.

## 2. Soft equivariance residual

Sample a group element \(g\) (rotation \(\varphi\), optional reflection) and a
feature-space map \(\rho(g)\) on student embeddings \(z\):

\[
L_{\mathrm{eq}} = \mathbb{E}\Bigl[\bar C\, \bigl\| \operatorname{stopgrad}\bigl(z(g\cdot x)\bigr) - \rho(g)\,z(x) \bigr\|^{p}\Bigr]
\]

**Default \(\rho\) (post Day-0 V2 fix):** fixed SO(2)/O(2) on the **first 2
feature dims** (`eq_action: so2_2d`). Remaining dims are unchanged. This cannot
learn identity for \(\varphi\not\equiv 0\).

**Legacy:** learned FiLM scale/shift (`eq_action: film`) — see collapse note below.

Ablations:

| Name | Meaning |
|------|---------|
| `vanilla_dino` | \(\lambda_{\mathrm{eq}}=0\) (no SIFESS term) |
| `sifess_full` | full \(L_{\mathrm{eq}}\) with \(\bar C\) gate |
| `no_c_gating` | same residual but \(\bar C := 1\) |
| `random_frame` | replace eigenframe angle by random \(\theta\) |

### Gradient / stop-grad choice

We **predict the framed embedding from the source** via \(\rho\):

- target = `stopgrad(z(g·x))`
- prediction = \(\rho(g)\,z(x)\)

Gradients flow through the **source student embed** and through \(\rho\) when it
is learned — not through the framed student embed. This is the BYOL-style
asymmetric residual: it blocks the trivial joint minimum
\(z(g\cdot x)\approx z(x)\) with \(\rho\approx\mathrm{Id}\).

### Guaranteed SO(2) plane energy

Fixed SO(2) is vacuous if the encoder dumps all mass into dims \(\ge 2\).
The student bottleneck therefore **packs** a dedicated 2D projection into the
first two embedding dims with fixed energy fraction \(\alpha\) (default
`plane_alpha=0.1`):

\[
z = \bigl[\sqrt{\alpha}\,\hat{u},\; \sqrt{1-\alpha}\,\hat{v}\bigr],
\quad \hat{u}\in\mathbb{R}^2,\;\hat{v}\in\mathbb{R}^{D-2}
\]

so plane energy \(=\alpha\) by construction. Under invariance,
\(\|z-R_\varphi z\|^2 = 2\alpha(1-\cos\varphi)\) stays \(O(\alpha)\) — a
sustained training signal. An optional soft floor
\(L_{\mathrm{plane}}=\mathrm{ReLU}(\tau-\mathbb{E}[z_0^2+z_1^2])\) remains as a
monitor (defaults \(\tau=0.05\), \(\lambda_{\mathrm{plane}}=0.1\)).

### Image-space action check

Training logs `tg_abs_mean` = \(\mathbb{E}|x - T_g x|\). A healthy run keeps this
clearly nonzero (warps are material); if it were ~0, \(L_{\mathrm{eq}}\) would be
uninformative for a different reason.

## 3. Day-0 V2 collapse diagnosis & fix

**Symptom (Kaggle Day-0 V2):** on `sifess_*` ablations, \(L_{\mathrm{eq}}\sim 5\cdot 10^{-4}\)
on epoch 1, then **0.0000** for epochs 2–8. Vanilla beat sifess on ID AUROC;
OOD brightness slightly favored sifess.

**Root cause (not a logging bug alone):**

1. **Learned FiLM \(\rho\)** was identity-initialized and stayed near identity.
2. **DINO pushes invariance**, so \(z(g\cdot x)\to z(x)\) within a few steps.
3. With **gradients on both** \(z(x)\) and \(z(g\cdot x)\), the residual
   \(\|z(g\cdot x)-\rho(g)z(x)\|\) has a trivial minimum: invariant embeds +
   \(\rho=\mathrm{Id}\) → \(L_{\mathrm{eq}}\to 0\).
4. Weak \(\lambda_{\mathrm{eq}}=0.5\) relative to \(L_{\mathrm{DINO}}\sim 7\)–8 made the
   collapse cheap.

(An earlier `.mean(dim=-1)` dilution also made printed \(L_{\mathrm{eq}}\) look like
0.0000 even when the vector-norm residual was small-but-nonzero; that dilution
was fixed separately. V2’s post-epoch-1 zeros were the invariance+identity
collapse above.)

**Fix (this revision):**

| Change | Why |
|--------|-----|
| Default \(\rho\) = fixed SO(2) on first 2 dims | Non-identity feature action for \(\varphi\neq 0\) |
| `stopgrad` on framed target | Block joint collapse; grads via source (+ρ if learned) |
| Packed SO(2) plane (\(\alpha=0.1\)) + \(L_{\mathrm{plane}}\) | Prevent dumping mass out of the SO(2) plane |
| Default \(\lambda_{\mathrm{eq}}=1.0\) (strong runs **2.0**) | Keep \(L_{\mathrm{eq}}\) a real training signal |
| Log \(\mathbb{E}\|x-T_g x\|\) | Confirm pixel action is material |

## 4. DINO student–teacher + total loss

We use a **DINO-style** student–teacher objective \(L_{\mathrm{DINO}}\)
(centering + sharpening; 2 global + 4 local crops) with a **single** ResNet
backbone family (ResNet-18 smoke / ResNet-50 default).

\[
L = L_{\mathrm{DINO}} + \lambda_{\mathrm{eq}} L_{\mathrm{eq}} + \lambda_{\mathrm{orth}} L_{\mathrm{orth}} + \lambda_{\mathrm{plane}} L_{\mathrm{plane}}
\]

Defaults: \(\lambda_{\mathrm{eq}}=1.0\), \(\lambda_{\mathrm{orth}}=0\) (pure SO(2)-on-2D),
\(\lambda_{\mathrm{plane}}=0.1\). Strong Kaggle script uses \(\lambda_{\mathrm{eq}}=2.0\).

Teacher weights follow EMA of the student (momentum default 0.996).

## 5. Why medical images

Radiographs have strong local orientation (ribs, vessels, fissures). Hard
equivariance to *lab-frame* rotations fights anatomy; soft equivariance to the
*image’s own* isophote frame respects structure where it exists and relaxes
where it does not — gated by \(C\).

## 6. What we do not claim here

No numerical leaderboard numbers are baked into this repo. Fill
`docs/EXPERIMENT_CARD.md` tables only after measured runs.
