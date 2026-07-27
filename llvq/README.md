# llvq — Leech Lattice Vector Quantization en Rust

Implémentation du papier **[LLVQ, arXiv:2603.11021](https://arxiv.org/abs/2603.11021)**
(van der Ouderaa, van Baalen, Whatmough, Nagel — Qualcomm AI Research, 2026) :
quantification vectorielle de poids de LLM sur le réseau de Leech Λ₂₄, état de
l'art à 2 bits/poids, sans codebook matérialisé.

Plan détaillé, gates de validation et provenance :
[`../research/llvq-rust-implementation-plan.md`](../research/llvq-rust-implementation-plan.md).

## État

| Phase | Contenu | Gate | Statut |
|---|---|---|---|
| 1 | `llvq-core` — Golay [24,12,8], Λ₂₄ (Eq. 4–5), couches | **G1** ✅ | fait |
| 2 | `llvq-search` — Adoul–Barth multi-couches (euclidien + angulaire) | G2 | à venir |
| 3 | Indexage bijectif hiérarchique | G3 | à venir |
| 4 | Validation source gaussienne (Table 3 : rétention 92,11 %) | G4 | à venir |
| 5 | Spherical GPTQ + pipeline LLM | G5 | à venir |
| 6 | Noyau CUDA fusé multi-couches | G6 | à venir |

Gate G1 (tout passe, `cargo test --release -- --include-ignored`, ~1,5 s) :
distribution des poids de Golay 1/759/2576/759/1, distance minimale 8,
auto-dualité, **nombre de baisers 196 560** et **|Shell(3)| = 16 773 120**
reproduits par énumération exhaustive où chaque vecteur compté est validé
individuellement par le prédicat d'appartenance, norme minimale 32, clôture
additive sur 10⁴ paires aléatoires.

## Stratégie de test LLM (phases 4+)

Du petit vers le gros, chaque étape ne servant qu'à dérisquer la suivante :

1. **Source gaussienne** — aucun modèle, cibles chiffrées de la Table 3.
2. **Qwen3-0.6B** — smoke test du pipeline (pas de chiffres de référence).
3. **Qwen3-4B** — premier modèle avec chiffres de référence dans le papier
   (Table 6) : c'est le juge de paix « petit modèle ».
4. **Llama-2 7B / Llama-3 8B** — comparaison finale aux tables du papier.

## Commandes

```bash
cargo test                                        # suite rapide (debug)
cargo test --release -- --include-ignored         # + Shell(3), ~25M vérifications
cargo clippy --all-targets
```

`llvq-core` n'a **aucune dépendance** : le cœur mathématique doit rester
auditable et reproductible (contexte souveraineté).
