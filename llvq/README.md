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
| 2 | `llvq-search` — Adoul–Barth multi-couches (euclidien + angulaire) | **G2** ✅ | fait (m ≤ 3) |
| 3 | Indexage bijectif hiérarchique | G3 | à venir |
| 4 | Validation source gaussienne (Table 3 : rétention 92,11 %) | G4 | à venir |
| 5 | Spherical GPTQ + pipeline LLM | G5 | à venir |
| 6 | Noyau CUDA fusé multi-couches | G6 | à venir |

Gate G1 (tout passe, `cargo test --release -- --include-ignored`, ~1,7 s) :
distribution des poids de Golay 1/759/2576/759/1, distance minimale 8,
auto-dualité, distinction des 4096 mots, **nombre de baisers 196 560** et
**|Shell(3)| = 16 773 120** reproduits par énumération exhaustive où chaque
vecteur compté est validé individuellement par le prédicat d'appartenance,
spot-checks Shell(4) (48 et 170 016), norme minimale 32, clôture additive.

La suite a été durcie par un audit adversarial multi-agents (mutation
testing) : le test `golay_stage_is_load_bearing` contient des sondes qui ne
sont rejetées **que** par l'étage Golay du prédicat — supprimer cet étage
fait échouer la suite (vérifié par mutation), ce qui n'était pas le cas de
la première version. Arithmétique exacte sur tout le domaine `[i32; 24]`
(norme en i128, add/neg vérifiés), `#![forbid(unsafe_code)]`.

Gate G2 (`llvq-search`) : recherche du plus proche voisin **exacte** sur
Shell(2), Shell(3) et leur union (métriques euclidienne *et* angulaire,
§3.1 du papier), validée contre la force brute (argmax sur les 196 560 puis
16,7 M points, énumérés par un troisième chemin de code lui-même épinglé à
la série thêta). Zéro énumération à la requête : maxima par classe en forme
close (réparation de parité par flip du min |xᵢ|), quantités par-codeword
en tables DP par chunks de 8 bits, élagage par borne supérieure. Débit
mesuré : **~7 300 requêtes/s/cœur** (au lieu de 507 en naïf) — l'objectif
10⁵ attend SIMD et l'itération triée de la Phase 2b ; à ce débit, encoder
Qwen3-4B prend ~12 min sur 32 cœurs, ce qui ne bloque pas G4/G5.

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
