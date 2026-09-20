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
learned map \(\rho(g)\) on student embeddings \(z\):

\[
L_{\mathrm{eq}} = \mathbb{E}\Bigl[\bar C\, \bigl\| z(g\cdot x) - \rho(g)\,z(x) \bigr\|^{p}\Bigr]
\]

Ablations:

| Name | Meaning |
|------|---------|
| `vanilla_dino` | \(\lambda_{\mathrm{eq}}=0\) (no SIFESS term) |
| `sifess_full` | full \(L_{\mathrm{eq}}\) with \(\bar C\) gate |
| `no_c_gating` | same residual but \(\bar C := 1\) |
| `random_frame` | replace eigenframe angle by random \(\theta\) |

## 3. DINO student–teacher + total loss

We use a **DINO-style** student–teacher objective \(L_{\mathrm{DINO}}\)
(centering + sharpening; 2 global + 4 local crops) with a **single** ResNet
backbone family (ResNet-18 smoke / ResNet-50 default).

\[
L = L_{\mathrm{DINO}} + \lambda_{\mathrm{eq}} L_{\mathrm{eq}} + \lambda_{\mathrm{orth}} L_{\mathrm{orth}}
\]

Defaults: \(\lambda_{\mathrm{eq}}=0.5\), \(\lambda_{\mathrm{orth}}=0.01\)
(set \(\lambda_{\mathrm{orth}}=0\) if the feature action is treated as pure SO(2) on 2D).

Teacher weights follow EMA of the student (momentum default 0.996).

## 4. Why medical images

Radiographs have strong local orientation (ribs, vessels, fissures). Hard
equivariance to *lab-frame* rotations fights anatomy; soft equivariance to the
*image’s own* isophote frame respects structure where it exists and relaxes
where it does not — gated by \(C\).

## 5. What we do not claim here

No numerical leaderboard numbers are baked into this repo. Fill
`docs/EXPERIMENT_CARD.md` tables only after measured runs.
