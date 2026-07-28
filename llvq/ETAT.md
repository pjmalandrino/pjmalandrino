# LLVQ — plan, réalisé, reste à faire

**Point d'entrée du projet.** Détails techniques : [`README.md`](README.md).
Contexte pour une session Claude Code : [`CLAUDE.md`](CLAUDE.md).
Plan d'origine : [`docs/llvq-rust-implementation-plan.md`](docs/llvq-rust-implementation-plan.md).

---

## 1. Le plan

### Pourquoi

Réduire le coût d'inférence LLM pour de la **souveraineté** : faire tenir de
plus gros modèles sur du matériel local. Le seul levier qui change la classe
de modèle qu'on peut charger, c'est le nombre de bits par poids.

| précision | taille d'un 70B | tient sur |
|---|---|---|
| FP16 (standard) | 140 Go | rien de local |
| 4 bits (AWQ/GPTQ) | 35 Go | 2× RTX 4090 |
| **2 bits (LLVQ)** | **18 Go** | **1× RTX 4090** |

Tout le monde sait descendre à 4 bits. À 2 bits, l'arrondi poids par poids
casse le modèle. Le papier LLVQ arrondit les poids **par paquets de 24**, en
projetant sur le réseau de Leech Λ₂₄ — la meilleure façon connue de disposer
des points en 24 dimensions (empilement de sphères optimal prouvé).

### Le papier

**[arXiv:2603.11021](https://arxiv.org/abs/2603.11021)** — van der Ouderaa,
van Baalen, Whatmough, Nagel · Qualcomm AI Research, mars 2026.
*Leech Lattice Vector Quantization for Efficient LLM Compression.*

Il revendique de battre AQLM, QuIP# et QTIP à 2 bits/poids. **Aucun code
publié** quatre mois après parution — c'est le trou qu'on comble.

### Pourquoi ce papier plutôt qu'un autre

Filtre appliqué lors de la veille : *fraîcheur × absence d'implémentation ×
levier réel sur la VRAM*. Contre-exemple instructif — TurboQuant (Google,
ICLR 2026) : cinq réimplémentations sérieuses sont apparues **en une semaine**,
dont une à 7 000 étoiles. Place prise, valeur d'ingénierie nulle. LLVQ : quatre
mois de silence, une seule tentative de reproduction dormante. La barrière est
technique (Leech en dimension 24 est difficile à implémenter), donc protectrice.

Détail de la veille et des 4 alternatives écartées :
[`docs/inference-cost-reduction-2026.md`](docs/inference-cost-reduction-2026.md).

### L'ouverture d'ingénierie

Elle est écrite noir sur blanc dans le papier (Annexe C) :

- leur noyau CUDA ne traite qu'**une seule couche (M = 3), « pour la simplicité »** ;
- il est **plus lent que QTIP**, et les auteurs déclarent l'optimisation bas
  niveau « largement orthogonale » à leur contribution ;
- le noyau **multi-couches**, celui qu'exige le régime 2 bits/poids (m ≤ 13),
  **n'existe nulle part**.

Autrement dit : la meilleure représentation de l'état de l'art, livrée avec un
noyau de démonstration que les auteurs eux-mêmes déclarent non optimisé.
L'écart entre les deux, c'est le projet.

### Les 7 gates

Chaque phase est verrouillée par un critère vérifiable. Deux sont des **points
de sortie** : si le critère tombe, on arrête et on le dit.

| Gate | Critère |
|---|---|
| G1 | Invariants publics de Λ₂₄ et du code de Golay |
| G2 | Recherche du plus proche voisin exacte vs force brute |
| G2b | Moteur générique m ≤ 13 vs série thêta |
| G3 | Bijection d'indexage, zéro collision |
| G4 | Rétention Shannon sur source gaussienne ≥ celle du papier |
| **G5** | **Bat QuIP#/QTIP en perplexité — sinon, point de sortie** |
| G6 | Noyau fusé ≥ 1,36× le FP16, multi-couches |

---

## 2. Ce qui est fait — 5 gates sur 7

4 035 lignes de Rust, 3 crates, 30 tests, zéro warning clippy, ~45 s de suite
complète. `llvq-core` et `llvq-search` ont **zéro dépendance** et
`forbid(unsafe_code)` — le cœur mathématique reste auditable, ce qui compte
dans un contexte souveraineté.

### G1 — fondations mathématiques ✅

`llvq-core` : code de Golay étendu [24,12,8] et construction entière de Λ₂₄.

Validé contre des constantes publiques, indépendantes du papier : distribution
des poids de Golay 1/759/2576/759/1, distance minimale 8, auto-dualité,
**nombre de baisers 196 560**, **|Shell(3)| = 16 773 120** — ces deux derniers
reproduits par énumération exhaustive où chaque vecteur compté est validé
individuellement par le prédicat d'appartenance.

**Un audit adversarial multi-agents a trouvé un défaut critique** : la première
version de la suite passait entièrement avec l'étage Golay du prédicat
supprimé. Toutes les énumérations construisaient leurs mots à partir de vrais
codewords, donc le prédicat dégénérait en simple filtre de somme. Corrigé par
des sondes valides en parité *et* en somme, rejetées uniquement par Golay —
la mutation échoue désormais. 12 constats appliqués au total.

### G2 — recherche exacte du plus proche voisin ✅

`llvq-search` : généralisation d'Adoul & Barth (1988) aux unions de couches,
avec les deux métriques du §3.1 (euclidienne pour le spherical shaping,
angulaire pour le shape–gain).

Zéro énumération à la requête : les maxima par classe ont une **forme close**,
et les quantités par-codeword passent par des tables DP en chunks de 8 bits.
Validé contre la force brute (argmax sur 196 560 puis 16,7 M points).
Débit : 507 → **7 278 requêtes/s/cœur** après trois passes d'optimisation.

### G2b — moteur générique m ≤ 13 ✅ *(le verrou central)*

Énumération programmatique des classes d'équivalence des 12 couches.
**Le test qui verrouille tout** : la formule de cardinalité doit reproduire les
coefficients thêta connus **et la somme cumulée exacte
N(13) = 280 974 212 784 720** (Table 1 du papier). Un verrou à 15 chiffres
qu'aucune contrainte fausse ne peut franchir.

Deux dérivations non triviales, testées :
- la condition de somme des classes **impaires** se réduit à `n₁+n₇+n₉ impair`
  au niveau classe (les signes forcés s'annulent mod 2) — donc maximiseur =
  appariement trié, exact par réarrangement, et **zéro bit** dépensé en signes ;
- la réparation de parité des classes **paires** est un *sacrifice au plus petit
  |xᵢ| du support avec retassement*, pas un flip en place (que j'avais d'abord
  implémenté à tort) — validé contre une référence DP exhaustive.

### G3 — indexage bijectif ✅

Bijection **point ↔ entier ≤ N(13)** sur 280 974 milliards de points, sans
matérialiser le codebook : **48 bits par bloc de 24 poids**, soit exactement
2 bits/poids. C'est la contribution (i) du papier.

Le pire mode de défaillance d'un codec est la collision silencieuse — elle
corrompt des poids en passant tous les tests de qualité. Verrouillée des deux
côtés : aller-retour **exhaustif** sur Shell(2) (196 560 points, zéro
collision, injectivité et surjectivité prouvées), **2 millions** d'indices
aléatoires sur les 2⁴⁸ points, frontières de classes, gagnants du moteur.

### G4 — validation qualité sur source gaussienne ✅

Protocole du §4 du papier, 20 000 blocs, sans aucun LLM :

| méthode | bits/dim | MSE | rétention Shannon |
|---|---|---|---|
| Lloyd–Max scalaire optimal | 1,000 | 0,3634 | 73,0 % |
| papier, spherical shaping | 2,000 | — | 89,14 % |
| papier, shape–gain | 2,000 | — | 92,11 % |
| **notre spherical shaping (β\* = 0,350)** | **1,9999** | **0,0775** | **92,23 %** |
| notre shape–gain, gain 2 bits | 2,0832 | 0,0670 | 93,62 % |
| Shannon | 2,000 | 0,0625 | 100 % |

**On atteint et dépasse la rétention du papier.** L'argument de solidité est
structurel : chaque score est atteint par un point matérialisé du codebook,
vérifié membre du réseau — un bug moteur ne pourrait que *dégrader* ces
chiffres, jamais les gonfler.

⚠️ **Réserve honnête** : la colonne MSE de la Table 3 telle que transcrite du
PDF (0,1084) est incohérente avec sa propre colonne SQNR (1,798 ⇒ MSE ≈ 0,0845).
Le PDF a un encodage de police décalé qui corrompt les chiffres de tableaux.
L'ancre auto-cohérente est la **rétention**. À revérifier visuellement sur le
PDF original avant de communiquer ces chiffres.

---

## 3. Reste à faire

### Préalable — décision matérielle *(à trancher, bloquant pour G6)*

Le plan prévoyait un noyau **CUDA**. Ça ne tourne pas sur Mac. Si le dev est
sur MacBook et le déploiement sur serveur NVIDIA, il faut choisir tôt :

| option | pour | contre |
|---|---|---|
| CUDA (`cudarc`) | perf max, tensor cores, cible serveur | rien sur Mac, verrou NVIDIA |
| `wgpu`/WGSL | AMD + Intel Arc + Apple, cohérent souveraineté | pas de tensor cores, outillage plus faible |

**Question ouverte : c'est quoi la machine cible réelle pour l'inférence ?**

### Phase 2c — performance de l'encodeur *(1 semaine)*

État : **~140 blocs/s/cœur** pour le moteur générique m ≤ 13.

| modèle | blocs | 4 cœurs | 12 cœurs (M-series) |
|---|---|---|---|
| Qwen3-0.6B | 21 M | 10 h | 3,5 h |
| Qwen3-4B | 150 M | 74 h | 25 h |
| Llama-3 8B | 292 M | 144 h | 48 h |

⚠️ C'est un coût **hors ligne, unique** — pas un coût d'inférence. Acceptable
pour le smoke test 0.6B, bloquant au-delà.

Goulot identifié (non profilé) : la boucle des classes impaires fait un **tri
de 24 éléments pour chacun des 4096 codewords**, à chaque bloc — ~450 000
opérations. Pistes par rapport gain/effort : (1) trier les classes par borne
décroissante et sortir tôt, (2) sélection partielle au lieu du tri complet,
(3) réutiliser le tri entre codewords voisins, (4) SIMD en dernier.

### Phase 5 — Spherical GPTQ et premier LLM *(2-3 semaines)* ⭐

**Le vrai jalon.** C'est ici qu'on sort du monde auto-vérifiable des maths
exactes, et c'est là qu'est le risque du projet.

1. Chargeur `safetensors` (première dépendance externe assumée).
2. Hessiennes par couche `H = AᵀA/N`. Le papier utilise 6 100 séquences de
   DCLM-edu. Algèbre : `faer` (pur Rust, pas de dépendance Fortran/BLAS).
3. **Algorithme 1 du papier** : blocs de 24 canaux d'entrée, gauche→droite,
   Cholesky de `H⁻¹`, reset de gain `ṽ = ‖v‖₂·Q_dir(v/‖v‖₂)`, propagation du
   résidu sur les colonnes non traitées.
4. Progression **petit → gros** : Qwen3-0.6B (smoke test) → **Qwen3-4B** (le
   plus petit modèle avec des chiffres de référence, Table 6 du papier) →
   Llama-2 7B / Llama-3 8B.
5. Évaluation : perplexité WikiText-2 à 4096, MMLU, CSR — **plus un benchmark
   métier d'extraction documentaire**. Cf.
   [arXiv:2607.08734](https://arxiv.org/abs/2607.08734) : perplexité et
   exactitude restent stables pendant que les réponses individuelles changent.

> **G5 est un point de sortie.** Si LLVQ ne bat pas QuIP#/QTIP sur Qwen3-4B,
> toute la thèse tombe. Il faut alors s'arrêter et le dire, pas optimiser un
> noyau pour une méthode qui ne tient pas ses promesses.

### Phase 6 — noyau fusé *(3-4 semaines)* ⭐

Là où est la contribution d'ingénierie. Deux objectifs, dans l'ordre :

1. **Reproduire** le noyau mono-couche des auteurs. Repère Table 7 (matvec
   4096×4096) : FP16 = 16,13 µs, leur LLVQ fusé = 11,194 µs → **1,36–1,48×**.
2. **Dépasser** : noyau **multi-couches** (inexistant, conditionne le régime
   2 bits/poids réel) et franchir la barre QTIP que les auteurs n'ont pas visée.

### Phase 7-8 — intégration et portabilité *(4-5 semaines)*

Intégration `mistral.rs` ou `candle`, format de sérialisation, CLI. Puis chemin
`wgpu` et CPU SIMD.

> **Gate final G7, la seule question qui compte :** *est-ce qu'un modèle qui
> n'entrait pas entre maintenant, sur le matériel réel, en restant utilisable ?*

---

## 4. Risques

| risque | probabilité | parade |
|---|---|---|
| Chiffres du papier non reproductibles sur LLM | moyenne | G5 avant tout investissement noyau |
| Qualcomm publie son code | faible | 4 mois de silence ; le noyau multi-couches + l'intégration gardent leur valeur |
| Ne pas battre QTIP en vitesse | moyenne | les auteurs non plus ; repli documenté acceptable |
| Collision d'indexage silencieuse | faible mais **critique** | G3, jamais relâché |
| Perf encodeur bloquante au-delà du 4B | élevée | Phase 2c, goulot déjà identifié |

## 5. Leçon de méthode

L'audit adversarial a montré qu'une suite de tests verte peut ne rien prouver.
**Avant de déclarer un gate vert : muter le code et vérifier que la suite
échoue.** Un test qui passe sur du code cassé ne vaut rien.
