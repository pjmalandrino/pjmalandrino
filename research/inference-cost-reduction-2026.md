# Réduction du coût d'inférence — veille & candidats à implémenter

**Date de la veille : juillet 2026**
**Objectif : faire tenir des modèles plus gros sur du matériel local (souveraineté), à qualité constante.**

---

## 0. Avertissement méthodologique

Cette veille a été réalisée depuis un environnement dont la politique réseau bloque
`arxiv.org`, `huggingface.co`, `openreview.net` et `semanticscholar.org` (403 sur le proxy
d'egress). Conséquence directe sur le niveau de confiance :

| Source | Vérifié ? |
|---|---|
| Existence, nom, date et description des dépôts GitHub | **Oui** — API GitHub interrogée directement |
| Titres, numéros arXiv, dates de soumission | Probable — issus des résultats de recherche, non lus sur arXiv |
| Chiffres de performance (× speedup, perplexité, tok/s) | **Non vérifiés** — repris des résumés, à revalider en lisant le PDF |

**Premier geste avant tout code : relire les PDF depuis un poste non filtré** et confirmer les
chiffres du tableau de la section 3. Les numéros arXiv donnés permettent d'y aller directement.

---

## 1. Où est réellement l'argent

Sur une inférence locale mono-utilisateur ou petit-batch (le cas « souveraineté »), le
goulot n'est presque jamais le calcul : c'est **la mémoire et la bande passante mémoire**.
Décomposition du budget VRAM :

```
VRAM = poids du modèle  +  KV cache  +  activations
        └─ dominant en    └─ dominant en    └─ négligeable
           batch 1,           contexte long     hors prefill
           contexte court     (>32k tokens)
```

D'où quatre leviers, par ordre de rendement décroissant pour l'objectif « modèle plus gros
sur la même machine » :

1. **Quantification des poids** — le seul levier qui change la classe de modèle qu'on peut
   charger. Passer de 4 bits à 3 bits fiables, c'est ~25 % de VRAM en moins, soit un 70B qui
   entre là où seul un 50B entrait.
2. **Compression / sparsification du KV cache** — décisif dès qu'on vise du contexte long
   (RAG documentaire, analyse de corpus). C'est le levier qui explose en 2026.
3. **Offload MoE intelligent (CPU/GPU/NVMe)** — permet de faire tourner des modèles dont les
   poids ne tiennent *pas du tout* en VRAM. Le levier le plus spectaculaire, le plus
   sensible à l'ingénierie système.
4. **Décodage spéculatif** — améliore la latence, pas la capacité mémoire. Hors scope
   principal ici, et le champ est déjà très outillé (EAGLE-3, TorchSpec).

---

## 2. Le piège à éviter : TurboQuant

TurboQuant (Google, ICLR 2026 — quantification KV cache 3 bits, *data-oblivious*, sans
calibration) est **le papier le plus médiatisé de l'année** sur le sujet, et c'est
précisément pour ça qu'il faut s'en méfier comme sujet d'implémentation.

État constaté sur GitHub (juillet 2026) :

| Dépôt | Étoiles | Créé |
|---|---|---|
| `TheTom/turboquant_plus` | ~7 000 | 25 mars 2026 |
| `0xSero/turboquant` (kernels Triton + vLLM) | ~1 700 | 25 mars 2026 |
| `scrya-com/rotorquant` (revendique de le battre) | ~1 040 | 26 mars 2026 |
| `tonbistudio/turboquant-pytorch` | ~1 034 | 25 mars 2026 |
| `mitkox/vllm-turboquant` | ~610 | 25 mars 2026 |
| `AmesianX/TurboQuant` (portage llama.cpp) | ~92 | 29 mars 2026 |

Le papier est sorti et **cinq réimplémentations sérieuses existaient sous une semaine**,
plus une discussion d'intégration ouverte sur `ggml-org/llama.cpp` (#20969). Il n'y a plus
de valeur d'ingénierie à créer là : la place est prise, et un concurrent (`rotorquant`)
prétend déjà faire mieux.

**Leçon transposable** : le signal « papier fraîchement publié » ne suffit pas. Le bon
filtre est *fraîcheur × absence d'implémentation × levier réel sur la VRAM*. Le reste de ce
document applique ce filtre.

---

## 3. Shortlist

Classement par ratio (valeur pour la souveraineté) / (effort d'ingénierie), en tenant compte
de l'espace laissé libre.

### 3.1 — HARP : rotations apprises en remplacement du Hadamard fixe

- **arXiv** : 2605.29843 (mai 2026)
- **Titre** : *HARP: Hadamard-Preconditioned Adaptive Rotation Processor for Extreme LLM Quantization*
- **Code** : `brain-lab-research/HARP` — **0 étoile**, créé le 27 mai 2026, dernier commit
  2 juillet 2026. Du code de recherche, pas un produit.

**L'idée.** Toutes les méthodes PTQ modernes (QuIP#, QTIP, QuaRot, SpinQuant) reposent sur
la *incoherence processing* : on multiplie les poids par une transformée de Hadamard
randomisée (RHT) pour disperser les outliers avant quantification. Cette rotation est
**fixe et aveugle** — la même pour toutes les couches, tous les modèles, tous les
quantiseurs. HARP la remplace par une rotation **apprise sur les données de calibration**,
paramétrée comme un produit d'étages block-orthogonaux « papillon » (structure FFT), donc
peu coûteuse à appliquer. Elle s'initialise sur la RHT à une permutation près : le pire cas
est donc de retrouver la performance actuelle.

**Pourquoi ça vaut le coup.** C'est le levier n°1 (poids), sur la plage 2–4 bits, sur des
modèles de 1B à 70B. Les chiffres annoncés (128 tok/s contre 61 tok/s en FP16, gain de
perplexité et de précision zero-shot contre RHT fixe) sont *à vérifier*, mais la logique est
solide : rendre adaptatif un composant jusqu'ici arbitraire.

**Le travail d'ingénieur.** Le papier fournit l'algorithme ; il manque tout l'aval.
1. Reproduire le fit des rotations sur un modèle public (Qwen3, Llama, Mistral) et vérifier
   qu'on bat bien RHT à budget de bits égal.
2. Écrire le kernel de rotation papillon (Triton ou CUDA) — c'est là qu'est la vraie valeur,
   parce qu'une rotation apprise mal implémentée annule son propre gain.
3. Sérialiser au format GGUF / exposer via vLLM, sinon personne ne s'en sert.
4. Gérer les dimensions non-puissances-de-2 (schedules mixed-radix), point que le papier
   traite explicitement et que les implémentations naïves ratent.

**Effort** : 3–6 semaines. **Risque** : moyen — les gains peuvent fondre après quantification
réelle. **Place libre** : oui, largement.

---

### 3.2 — CoX-MoE : co-exécution CPU (AMX) / GPU pour les MoE

- **arXiv** : 2605.17889 (mai 2026)
- **Titre** : *CoX-MoE: Coalesced Expert Execution for High-Throughput MoE Inference with AMX-Enabled CPU-GPU Co-Execution*
- **Code** : **aucun dépôt GitHub.** Recherche par nom exact : 0 résultat.

**L'idée.** Un modèle MoE n'active qu'une fraction de ses experts par token, mais tous les
poids doivent être quelque part. L'approche classique garde les experts chauds en VRAM et
les froids en RAM, avec un transfert PCIe à chaque miss — le transfert domine. CoX-MoE
propose de **calculer les experts froids directement sur le CPU**, en exploitant les
instructions Intel AMX (multiplication matricielle native sur Xeon Sapphire Rapids et
suivants), et de *coalescer* les exécutions d'experts pour amortir les coûts fixes.

**Pourquoi c'est le meilleur candidat « souveraineté ».** Le parc on-premise français typique,
ce n'est pas un cluster H100 : c'est un bi-Xeon récent avec beaucoup de RAM et une ou deux
cartes de milieu de gamme. AMX est présent et massivement sous-exploité sur ce matériel. Un
runtime qui sait faire tourner les experts froids sur le CPU au lieu de les rapatrier
transforme un serveur généraliste en machine à MoE — c'est exactement l'écart entre « on ne
peut pas héberger ce modèle » et « on l'héberge ».

**Le travail d'ingénieur.** Tout est à faire, et c'est du système, pas du ML :
1. Kernel expert GEMM sur AMX (intrinsics `_tile_*`, ou via oneDNN) en INT8/BF16.
2. Politique de placement chaud/froid et coalescing du batch d'experts.
3. Pipeline asynchrone GPU ↔ CPU pour recouvrir calcul et transfert.
4. Benchmark honnête contre l'existant : `llama.cpp` avec `--n-cpu-moe`, ktransformers,
   et le récent `JustVugg/colibri` (~20 000 étoiles, créé le 1er juillet 2026, qui fait
   tourner un MoE de 744B sur 25 Go de RAM en streamant les experts depuis le disque).

**Effort** : 6–10 semaines, compétences bas niveau requises. **Risque** : faible sur la
faisabilité, élevé sur la reproduction exacte des chiffres. **Place libre** : totale.

> Le voisinage immédiat, si CoX-MoE déçoit à la lecture : *Efficient CPU-GPU Collaborative
> Inference for MoE-based LLMs on Memory-Limited Systems* (arXiv 2512.16473, ASP-DAC 2026),
> *MoBiLE* (2510.12357, aucun dépôt trouvé) et *Dynamic Expert Quantization* (2511.15015),
> qui garde les experts à fort trafic en haute précision et les autres en fallback basse
> précision. Même levier, angles différents.

---

### 3.3 — Quantification vectorielle par réseau de Leech

- **arXiv** : 2603.11021 (mars 2026)
- **Titre** : *Leech Lattice Vector Quantization for Efficient LLM Compression*
- **Code** : **aucune implémentation LLM sérieuse.** Les résultats GitHub sur « Leech
  lattice » sont des projets de compression généraliste ou des curiosités mathématiques.

**L'idée.** La quantification scalaire perd par construction : quantifier chaque poids
indépendamment ignore la structure du vecteur. La quantification vectorielle sur réseau
exploite le fait qu'en dimension 24, le réseau de Leech est l'empilement de sphères optimal
prouvé — c'est le codebook théoriquement le meilleur possible à cette dimension, avec un
gain de granularité connu (~1,03 dB sur la quantification scalaire).

**Pourquoi c'est un beau sujet d'ingénieur.** Contrairement à un papier de ML flou, ici la
partie mathématique est **entièrement spécifiée et publique depuis quarante ans** :
l'algorithme de décodage de Conway–Sloane pour Λ24 est documenté, exact, déterministe. Le
travail n'est pas de deviner ce que voulaient dire les auteurs, c'est d'écrire un décodeur
rapide. C'est un profil de tâche à faible risque d'échec et à forte valeur : si le décodage
tient en quelques nanosecondes par vecteur sur GPU, on a un quantiseur de poids meilleur que
l'état de l'art à 2–3 bits.

**Le travail d'ingénieur.**
1. Décodeur Leech vectorisé sur GPU (le point dur : il faut qu'il soit *rapide*, sinon la
   dequantification tue le gain mémoire).
2. Combinaison avec l'incoherence processing (Hadamard ou HARP en amont) — les deux se
   composent naturellement.
3. Comparaison frontale avec QTIP (`Cornell-RelaxML/qtip`), qui est le SOTA sortant et
   utilise un treillis plutôt qu'un réseau.

**Effort** : 4–8 semaines. **Risque** : faible côté correction, réel côté vitesse de
décodage. **Place libre** : oui.

---

### 3.4 — Attention Editing : conversion GQA → MLA sur modèles déjà post-entraînés

- **arXiv** : 2604.05688 (avril 2026)
- **Titre** : *Attention Editing: A Versatile Framework for Cross-Architecture Attention Conversion*
- **Code** : non trouvé.

**L'idée.** MLA (Multi-head Latent Attention, l'attention de DeepSeek) compresse le KV cache
d'un ordre de grandeur par rapport à GQA. Mais tous les modèles ouverts occidentaux
(Llama, Qwen, Mistral) sont en GQA. TransMLA et MHA2MLA ont montré qu'on pouvait convertir
*a posteriori* — mais sur les modèles de base uniquement. Attention Editing traite
l'attention cible comme un module remplaçable appris, ce qui étend la conversion aux
modèles **déjà instruits ou entraînés au raisonnement** — c'est-à-dire ceux qu'on déploie
réellement.

**Pourquoi c'est stratégique.** C'est le seul levier de la liste qui change
l'*architecture* et non l'encodage. Le gain sur le KV cache se cumule avec la quantification
des poids et n'est pas plafonné par le même mur. Concrètement : servir un modèle instruit
souverain avec un contexte 4× plus long à VRAM constante.

**Le travail d'ingénieur.** Pipeline de conversion reproductible sur un modèle cible
(Mistral ou Qwen instruct), + évaluation de non-régression sérieuse — c'est là que ça se
joue, parce que le risque d'abîmer l'alignement ou les capacités de raisonnement lors de
l'édition est réel et c'est précisément ce que le papier prétend résoudre.

**Effort** : 4–6 semaines dont beaucoup d'évaluation. **Risque** : élevé (dégradation
silencieuse des capacités). **Place libre** : oui.

> Voisin à lire en même temps : *GQLA / TransGQLA* (arXiv 2605.15250, mai 2026), qui vise le
> compromis GQA-efficacité / MLA-compression avec adaptation au matériel.

---

### 3.5 — RaBitQCache : attention sparse à budget adaptatif

- **arXiv** : 2606.31519 (30 juin 2026, ICML'26) — **le plus frais de la liste**
- **Code** : `Sakuraaa0/RaBitQCache` — dépôt officiel, ~14 étoiles.

**L'idée.** Les méthodes d'attention sparse récupèrent les top-k tokens du KV cache avec un
budget *fixe*. RaBitQCache utilise une quantification binaire rotationnelle randomisée
(RaBitQ, une technique venue des bases vectorielles) pour estimer les poids d'attention en
arithmétique binaire-INT4. L'estimateur est **non biaisé avec borne d'erreur prouvée**, ce
qui autorise une récupération **top-p adaptative** : le budget de tokens s'ajuste à la
sparsité réelle de l'attention au lieu d'être deviné.

**Statut particulier.** Le code officiel existe — donc pas « en attente d'implémentation »
au sens strict. Mais 14 étoiles, aucune intégration moteur, et c'est du code de papier ICML.
Le travail disponible n'est pas la réimplémentation, c'est le **portage vers vLLM / SGLang /
llama.cpp**, où il n'y a rien. Et l'auteur amont a un intérêt direct à voir arriver cette PR.

**Effort** : 2–4 semaines (le plus court de la liste, parce qu'on part de code qui tourne).
**Risque** : faible. **Place libre** : oui côté intégration, non côté algo.

> Concurrence directe dans le même créneau, à arbitrer à la lecture : *UNIQUE* (2605.27740,
> top-k sparse universel à granularité page KV), *Fluxion* (2605.07719, sparse hybride avec
> parallélisme CPU-GPU), *HiLS* (2607.02980, juillet 2026, attention hiérarchique
> extrapolant à 64× la longueur d'entraînement), *OSCAR* (2605.17757, rotation covariante
> pour KV 2 bits).

---

## 4. Tableau de décision

| # | Papier | arXiv | Levier | Code amont | Effort | Impact souveraineté |
|---|---|---|---|---|---|---|
| 3.2 | **CoX-MoE** | 2605.17889 | Offload MoE | **Aucun** | 6–10 sem | ⭐⭐⭐⭐⭐ |
| 3.1 | **HARP** | 2605.29843 | Poids | Squelette (0 ★) | 3–6 sem | ⭐⭐⭐⭐ |
| 3.3 | **Leech VQ** | 2603.11021 | Poids | Aucun | 4–8 sem | ⭐⭐⭐⭐ |
| 3.4 | **Attention Editing** | 2604.05688 | Architecture / KV | Aucun | 4–6 sem | ⭐⭐⭐⭐ |
| 3.5 | **RaBitQCache** | 2606.31519 | KV cache | Officiel (14 ★) | 2–4 sem | ⭐⭐⭐ |
| — | ~~TurboQuant~~ | — | KV cache | 6+ impls, 12k ★ | — | **Saturé** |

---

## 5. Séquencement proposé

**Vague 0 — une semaine, avant tout code.**
Construire le banc d'essai. Sans lui, aucune des cinq pistes n'est évaluable et on ne saura
pas distinguer un vrai gain d'un artefact de mesure.
- Matériel cible figé et documenté (le vrai parc, pas une H100 louée).
- Métriques : VRAM pic, tok/s prefill, tok/s decode, perplexité **et** un benchmark métier
  — pour ce contexte, une tâche d'extraction documentaire réelle plutôt qu'un MMLU. Les
  métriques classiques masquent les régressions : c'est exactement le propos de *The
  Illusion of Equivalency in Quantization* (arXiv 2607.08734, juillet 2026), qui montre que
  perplexité et exactitude restent stables alors que les réponses individuelles changent
  significativement. À lire avant de définir le protocole.
- Baselines figées : FP16, GGUF Q4_K_M, AWQ/GPTQ 4 bits.

**Vague 1 — RaBitQCache (3.5).** Le plus court chemin vers un résultat publiable, sur du
code qui tourne déjà. Sert à roder le banc d'essai et à établir la crédibilité de la
démarche avec un risque minimal.

**Vague 2 — au choix selon le profil de l'équipe :**
- profil système bas niveau → **CoX-MoE (3.2)**, le pari le plus rentable ;
- profil ML/quantification → **HARP (3.1)** ou **Leech VQ (3.3)**, qui se composent et
  peuvent être menés ensemble (HARP en amont, Leech en aval du même pipeline).

**Vague 3 — Attention Editing (3.4)**, une fois qu'on a un protocole d'évaluation
suffisamment robuste pour détecter une dégradation subtile des capacités. Le faire plus tôt,
c'est prendre le risque de conclure à tort.

---

## 6. Sources

Papiers :
- [HARP — 2605.29843](https://arxiv.org/abs/2605.29843)
- [CoX-MoE — 2605.17889](https://arxiv.org/pdf/2605.17889)
- [Leech Lattice VQ — 2603.11021](https://arxiv.org/pdf/2603.11021)
- [Attention Editing — 2604.05688](https://arxiv.org/pdf/2604.05688)
- [RaBitQCache — 2606.31519](https://arxiv.org/abs/2606.31519)
- [UNIQUE — 2605.27740](https://arxiv.org/abs/2605.27740)
- [Fluxion — 2605.07719](https://arxiv.org/abs/2605.07719)
- [HiLS — 2607.02980](https://arxiv.org/abs/2607.02980)
- [The Illusion of Equivalency in Quantization — 2607.08734](https://arxiv.org/abs/2607.08734)
- [GQLA — 2605.15250](https://arxiv.org/html/2605.15250v1)
- [OSCAR — 2605.17757](https://arxiv.org/pdf/2605.17757)
- [MoBiLE — 2510.12357](https://arxiv.org/html/2510.12357)
- [Dynamic Expert Quantization — 2511.15015](https://arxiv.org/abs/2511.15015)
- [CPU-GPU Collaborative MoE Inference — 2512.16473](https://arxiv.org/abs/2512.16473)
- [Token Sparse Attention — 2602.03216](https://arxiv.org/abs/2602.03216)
- [D2Quant — 2602.02546](https://arxiv.org/html/2602.02546v2)
- [QTIP (référence sortante) — 2406.11235](https://arxiv.org/abs/2406.11235)

Dépôts et écosystème :
- [Sakuraaa0/RaBitQCache](https://github.com/Sakuraaa0/RaBitQCache)
- [brain-lab-research/HARP](https://github.com/brain-lab-research/HARP)
- [Cornell-RelaxML/qtip](https://github.com/Cornell-RelaxML/qtip)
- [JustVugg/colibri](https://github.com/JustVugg/colibri)
- [NVIDIA/kvpress](https://github.com/NVIDIA/kvpress)
- [ikawrakow/ik_llama.cpp](https://github.com/ikawrakow/ik_llama.cpp)
- [Discussion TurboQuant sur llama.cpp](https://github.com/ggml-org/llama.cpp/discussions/20969)
