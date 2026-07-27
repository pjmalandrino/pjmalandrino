# LLVQ en Rust — plan d'implémentation

> **Papier de référence : [Leech Lattice Vector Quantization for Efficient LLM Compression — arXiv:2603.11021](https://arxiv.org/abs/2603.11021)**
> van der Ouderaa, van Baalen, Whatmough, Nagel — Qualcomm AI Research, mars 2026.
> Miroir HTML : <https://arxiv.org/html/2603.11021v1> · PDF : <https://arxiv.org/pdf/2603.11021>

Lectures d'accompagnement obligatoires :
- [QuIP# — arXiv:2402.04396](https://arxiv.org/abs/2402.04396) : le codebook E8P, la baseline à battre, et la référence pour le noyau de déquantification.
- [PVQ for LLMs — arXiv:2410.16926](https://arxiv.org/abs/2410.16926) : même équipe, prédécesseur direct de LLVQ, c'est de là que vient la décomposition shape–gain.
- [QTIP — arXiv:2406.11235](https://arxiv.org/abs/2406.11235) : l'autre SOTA, approche treillis.
- Conway & Sloane, *Sphere Packings, Lattices and Groups*, ch. 10 et 12 — la construction de Λ₂₄ et le décodeur.

---

## 0. Statut de ce plan — à lire avant de commencer

**Je n'ai pas pu lire le PDF.** L'environnement bloque `arxiv.org`, ses miroirs et les
dépôts tiers. L'architecture décrite en §1 est **reconstituée par recoupement** de sources
secondaires. Elle est cohérente et probablement juste dans les grandes lignes, mais :

| Élément | Confiance |
|---|---|
| Décomposition shape–gain + GPTQ sphérique | Élevée — confirmée par plusieurs sources, et c'est la continuité directe de PVQ |
| Construction via code de Golay étendu, décodage type hexacode | Élevée — c'est l'état de l'art connu pour Λ₂₄ |
| Trois contributions (indexage / recherche angulaire / noyau parallèle) | Élevée — formulation reprise mot pour mot dans plusieurs résumés |
| **Allocation exacte des bits, choix des couches, bijection d'indexage** | **Nulle — à transcrire du papier** |
| **Chiffres de perplexité et tableaux comparatifs** | **Nulle — non vérifiés** |

La Phase 0 existe précisément pour combler cet écart. **Ne pas la sauter.** Et chaque phase
ci-dessous est verrouillée par un *gate* mathématique vérifiable indépendamment du papier :
c'est la parade à mon accès dégradé.

**Correction d'estimation** : j'avais annoncé 4–8 semaines dans la note de veille. C'était
pour un portage PyTorch. En Rust, avec noyau GPU et intégration moteur, compter
**11–15 semaines**. L'écart n'est pas du gaspillage — il achète un artefact déployable
plutôt qu'un notebook.

---

## 1. Ce que fait LLVQ (reconstitution)

### 1.1 Le problème que ça résout

La quantification scalaire quantifie chaque poids isolément et bute sur la borne
débit-distorsion. La quantification vectorielle encode des blocs conjointement et franchit
cette borne — mais elle bute sur le stockage du codebook.

C'est là qu'est le nœud, et il faut le comprendre pour saisir pourquoi ce papier compte :

```
Dimension 8  (E8, QuIP#)  →  2 bits/dim → 2^16 entrées → table de 256 Ko → tient en L1/shared
Dimension 24 (Leech)      →  2 bits/dim → 2^48 entrées → 280 To         → impossible
```

**QuIP# a choisi E8 pour cette raison exacte.** Leech est un meilleur empilement mais
personne ne savait s'en servir sans matérialiser le codebook. LLVQ contourne le mur en
calculant les points du réseau *à la volée* à partir de la structure algébrique du code de
Golay étendu, au lieu de les stocker.

### 1.2 Les trois briques

**(i) Indexage sans codebook matérialisé.** Bijection `bitstring ↔ point du réseau`, obtenue
en exploitant la structure de Λ₂₄ construit sur le code de Golay [24,12,8]. Un point se
reconstruit à partir d'un mot de Golay (12 bits d'information) plus des bits de coset et
d'échelle, via des congruences modulo 2, 4 et 8.

**(ii) Recherche angulaire sur une union de couches.** Une fois le gain factorisé, le bon
critère n'est plus la distance euclidienne mais **l'angle** : on cherche le point du réseau
qui maximise la similarité cosinus avec le vecteur cible, en balayant plusieurs couches
(*shells*) de norme différente au lieu d'une seule.

**(iii) Noyau de déquantification parallélisable.** Reconstruction de points de Leech
*bornés sphériquement*, par arithmétique modulaire rapide. Point crucial : les modulos sont
des puissances de deux, donc **tout se ramène à des opérations bit à bit** — pas de table,
pas de branchement lourd, pas de synchronisation entre threads.

### 1.3 Le pipeline complet

```
Poids W (par blocs de 24)
      │
      ├─► gain  g = ‖w‖         → quantifié à part (scalaire, partagé par groupe)
      │
      └─► shape u = w/‖w‖       → recherche angulaire sur ∪ couches de Λ₂₄
                                 → point du réseau → indexage → bitstring
      │
      └─► GPTQ sphérique : propagation de l'erreur pondérée par la Hessienne,
          adaptée au cas shape–gain (colonne par colonne, comme GPTQ)
```

### 1.4 L'asymétrie qui rend le projet faisable

C'est le point le plus important du plan, et celui que j'avais mal formulé dans la note de
veille initiale :

| | Quand | Coût acceptable | Nature |
|---|---|---|---|
| **Encodeur** (recherche du plus proche point) | Hors ligne, une fois par modèle | Minutes à heures | Décodeur hexacode, complexe |
| **Décodeur** (index → point) | **Chemin chaud, à chaque GEMM** | Nanosecondes | Bit-twiddling, trivial |

On paie la complexité une seule fois, à la quantification. Le chemin critique de
l'inférence ne fait que du décalage et du masquage. **C'est pour ça que ça peut marcher, et
c'est pour ça que Rust est un bon choix** : l'encodeur profite du système de types pour un
algorithme combinatoire délicat, le décodeur compile en SIMD propre.

---

## 2. Décisions techniques Rust

### 2.1 Découpage en workspace

```
llvq/
├── llvq-core/       # Golay, Λ₂₄, indexage, encode/decode. #![no_std], zéro dépendance.
├── llvq-quant/      # shape–gain, GPTQ sphérique, Hessiennes. faer.
├── llvq-kernels/    # déquantification : CUDA (cudarc) + SIMD CPU + wgpu.
├── llvq-format/     # sérialisation, extension GGUF / safetensors.
├── llvq-engine/     # intégration mistral.rs / candle.
├── llvq-cli/        # quantize | eval | bench
└── llvq-bench/      # banc d'essai, perplexité, tok/s, VRAM
```

`llvq-core` sans dépendance et `no_std` n'est pas de la coquetterie : c'est ce qui rend le
cœur testable exhaustivement, portable, et auditable — ce qui compte dans un contexte
souveraineté.

### 2.2 Choix de crates, avec les arbitrages

| Besoin | Choix | Pourquoi, et ce qu'on écarte |
|---|---|---|
| Algèbre linéaire (Hessiennes, Cholesky) | **`faer`** | Pur Rust, perf comparable à LAPACK. Écarte `ndarray-linalg` qui traîne une dépendance Fortran/BLAS — mauvais pour la reproductibilité de build. |
| GPU CUDA | **`cudarc`** + noyau CUDA C via NVRTC | Bindings driver minces, compilation du noyau à l'exécution. |
| GPU portable | **`wgpu`** + WGSL, en phase 7 | Fait tourner sur AMD, Intel Arc, Apple. Vrai argument souveraineté : pas de dépendance NVIDIA. |
| SIMD CPU | **`pulp`** ou `std::simd` | `pulp` dispatche AVX-512/AVX2/NEON sans nightly. |
| Moteur d'inférence | **`mistral.rs`**, sinon `candle` | Rust natif, gère déjà plusieurs formats quantifiés. |
| Tests de propriété | **`proptest`** | Idéal pour la bijectivité d'indexage (§4, gate G3). |
| Micro-bench | **`criterion`** | |

### 2.3 Un point à ne pas se raconter d'histoires

**Le noyau GPU ne sera pas du Rust.** Aucune chaîne Rust-vers-GPU ne donne aujourd'hui les
performances d'un noyau CUDA écrit à la main quand on vise le niveau de QTIP. `rust-gpu` et
Rust-CUDA existent mais ne sont pas au niveau. Le plan assume donc : **noyau en CUDA C,
piloté depuis Rust par `cudarc`**. Tout le reste — encodeur, pipeline, format, CLI, moteur —
est en Rust. Si la contrainte « 100 % Rust » est non négociable, la voie est `wgpu` + WGSL,
en acceptant de perdre l'accès aux tensor cores.

---

## 3. Les phases

### Phase 0 — Spécification depuis le papier · 3 à 5 jours

Rien ne commence avant ça. Depuis un poste **non filtré**, lire 2603.11021 et en extraire un
document de spec qui fixe :

1. La construction exacte de Λ₂₄ retenue (Construction A/B sur Golay ? autre normalisation ?)
   et l'échelle utilisée.
2. L'algorithme de recherche du plus proche point, et ce que les auteurs y ont *ajouté*.
3. **La bijection d'indexage, en détail** — c'est le cœur de la contribution et le point le
   plus susceptible d'être sous-spécifié dans le papier.
4. La liste des couches retenues et le budget de bits à 2, 3 et 4 bpw.
5. La quantification du gain : combien de bits, quel groupement, partagé sur quoi.
6. Le GPTQ sphérique : ordre des colonnes, amortissement, damping de la Hessienne.
7. Les tableaux de résultats sur Llama-2 7B et Llama-3 8B, à recopier comme cibles chiffrées.

> **Gate G0.** Les sept points sont-ils spécifiés sans ambiguïté ?
> **Non** → écrire aux auteurs avant d'engager quoi que ce soit. Qualcomm AI Research répond
> généralement. Un mail de trois jours d'attente coûte moins cher que trois semaines de
> reverse-engineering. Si pas de réponse sous deux semaines : **arrêter le projet** et
> basculer sur RaBitQCache (§3.5 de la note de veille), qui a du code officiel qui tourne.

---

### Phase 1 — Cœur mathématique · 1 à 2 semaines

`llvq-core`, sans dépendance : code de Golay étendu, construction de Λ₂₄, appartenance au
réseau, énumération des couches.

Représentation : mot de Golay en `u32` (24 bits utiles), point du réseau en `[i32; 24]`.
Encodage de Golay par matrice génératrice, syndrome par table de 2¹² — ça, ça tient en
mémoire, contrairement au codebook complet.

> **Gate G1 — invariants durs, non négociables.** Le réseau de Leech est l'un des objets les
> mieux caractérisés des mathématiques. Si un seul de ces tests échoue, l'implémentation est
> fausse ; s'ils passent tous, elle est presque certainement juste :
>
> | Test | Valeur attendue |
> |---|---|
> | Code de Golay : mots au total | 4096 |
> | Distribution des poids | 1 × 0, 759 × 8, 2576 × 12, 759 × 16, 1 × 24 |
> | Distance minimale | 8 |
> | Λ₂₄ : norme minimale | 4 |
> | **Nombre de baisers** (vecteurs de norme 4) | **196 560** |
> | Vecteurs de norme 6 | 16 773 120 |
> | Vecteurs de norme 8 | 398 034 000 |
> | Déterminant de la matrice de Gram | 1 (unimodulaire) |
> | Somme de deux points du réseau | ∈ réseau (clôture, sur 10⁶ tirages) |
>
> Ces nombres sont publics et vérifiables. **C'est le gate le plus rassurant du projet** :
> il valide le cœur sans dépendre de ma lecture du papier ni de la vôtre.

---

### Phase 2 — Encodeur : recherche du plus proche point · 1 à 2 semaines

Décodeur borné en distance de type Conway–Sloane / Vardy–Be'ery, via l'hexacode (code
[6,3,4] sur GF(4)) et le MOG. Puis extension à la **recherche angulaire sur union de
couches** telle que spécifiée en Phase 0.

> **Gate G2.**
> 1. Sur 10⁵ points tirés uniformément, le résultat coïncide avec une recherche exhaustive
>    restreinte au voisinage. Tolérance : **zéro écart** sur la version exacte.
> 2. Si le décodeur est borné en distance et non exact, mesurer le taux de sous-optimalité
>    et vérifier qu'il correspond à ce qu'annonce le papier.
> 3. Débit ≥ 10⁵ blocs/s/cœur — sinon quantifier un 70B devient déraisonnable.
>    À 70 milliards de poids, soit ~3×10⁹ blocs : 10⁵ blocs/s/cœur sur 32 cœurs ≈ 15 min.
>    Acceptable. Un ordre de grandeur en dessous, ça ne l'est plus.

---

### Phase 3 — Indexage bijectif · 1 semaine

`index → point` et `point → index`. C'est ici que se joue la contribution du papier.

**Contrôle de cohérence à faire avant de coder** — vérifier que le budget de bits et le
nombre de points représentables se rejoignent. Λ₂₄ étant unimodulaire, le nombre de points
dans une boule de rayon R vaut approximativement `V₂₄ · R²⁴` avec `V₂₄ = π¹²/12! ≈ 1,93×10⁻³`.
Pour couvrir 2⁴⁸ points (2 bits/dim) : `R²⁴ ≈ 1,4×10¹⁷`, soit `R ≈ 5,2`, donc des couches
jusqu'à la norme ≈ 27. Estimation asymptotique, grossière aux petits rayons — mais si la
spec issue de la Phase 0 donne un ordre de grandeur très différent, **c'est que la spec a
été mal lue**. Ce calcul de coin de table vaut une journée de débogage.

> **Gate G3 — bijectivité, par `proptest`.**
> 1. Aller-retour exact : `∀ i, index(point(i)) == i` sur 10⁷ tirages.
> 2. Aller-retour inverse : `∀ p ∈ réseau, point(index(p)) == p`.
> 3. **Absence de collision** : sur un sous-espace énumérable exhaustivement, l'application
>    est injective. Une collision, c'est de la corruption silencieuse de poids — le pire
>    mode de défaillance possible, il passe tous les tests de perplexité en dégradant
>    subtilement le modèle.
> 4. Tout index dans le budget de bits décode vers un point valide (pas de trou).

---

### Phase 4 — Pipeline de quantification · 2 semaines

`llvq-quant` : chargement safetensors, accumulation des Hessiennes par couche sur corpus de
calibration, shape–gain, GPTQ sphérique, sérialisation.

> **Gate G4 — le vrai juge de paix qualité.** Reproduire les chiffres du papier sur
> **Llama-2 7B et Llama-3 8B**, à 2 et 3 bpw, en perplexité WikiText-2 et C4.
> - Écart ≤ 0,05 de perplexité → validé, on continue.
> - Écart entre 0,05 et 0,2 → chercher l'erreur (ordre GPTQ, damping, groupement du gain).
> - Écart > 0,2, ou on ne bat pas QuIP#/QTIP → **point de sortie**. Toute la thèse du projet
>   est « LLVQ bat le SOTA ». Si ça ne se reproduit pas, il faut s'arrêter là et le dire,
>   pas optimiser un noyau pour une méthode qui ne tient pas ses promesses.
>
> Ajouter ici le benchmark métier — extraction documentaire — plutôt que d'attendre la fin.
> Cf. [*The Illusion of Equivalency in Quantization*, arXiv:2607.08734](https://arxiv.org/abs/2607.08734) :
> perplexité et exactitude restent stables pendant que les réponses individuelles changent.

---

### Phase 5 — Noyau de déquantification GPU · 2 à 3 semaines

Le passage obligé. Noyau CUDA C fusionné dans le GEMM (déquantifier en mémoire globale
puis relire annulerait tout le bénéfice), piloté par `cudarc`. Référence de structure :
le noyau E8P de QuIP#, et `cnygaard/glq` pour un exemple E8 fusionné en Triton.

> **Gate G5 — débit.**
> 1. **Le GEMM déquantifié doit rester borné par la bande passante mémoire, pas par le
>    calcul de déquantification.** Mesure directe : temps du GEMM LLVQ / temps du GEMM FP16
>    à même trafic mémoire. Au-delà de 1,3×, le noyau est le goulot.
> 2. tok/s en décodage batch 1, contre QTIP et contre GGUF Q4_K_M, **sur le matériel cible**.
> 3. Si on n'atteint pas 80 % du débit de QTIP : itérer une fois, puis livrer quand même en
>    documentant l'écart. Un quantiseur plus précis mais 20 % plus lent reste utile quand
>    l'objectif est de faire *entrer* le modèle.

---

### Phase 6 — Intégration moteur · 2 semaines

`llvq-engine` : type de quantification dans `mistral.rs`, format de sérialisation, chargement
de bout en bout, CLI.

> **Gate G6 — mesure de bout en bout sur le matériel réel, pas sur une carte louée.**
> VRAM pic, tok/s prefill et decode, perplexité, benchmark métier. La question à laquelle
> il faut répondre par oui ou non : **est-ce qu'un modèle qui n'entrait pas entre
> maintenant ?** C'est le seul critère qui compte pour l'objectif de départ.

---

### Phase 7 — Portabilité · optionnel, 2 à 3 semaines

Chemin `wgpu`/WGSL (AMD, Intel Arc, Apple) et chemin CPU SIMD via `pulp` pour les serveurs
Xeon sans GPU. À arbitrer seulement après G6 : c'est ce qui transforme le projet en brique
réellement souveraine, mais ça ne vaut rien si G5 n'est pas franchi.

---

## 4. Récapitulatif

| Phase | Durée | Gate | Sortie si échec |
|---|---|---|---|
| 0 — Spec depuis le papier | 3–5 j | G0 : 7 points spécifiés | Mailer les auteurs, puis abandonner |
| 1 — Golay + Λ₂₄ | 1–2 sem | G1 : invariants (196 560…) | Bug, pas d'abandon |
| 2 — Encodeur | 1–2 sem | G2 : exactitude + 10⁵ blocs/s | Bug, pas d'abandon |
| 3 — Indexage | 1 sem | G3 : bijectivité `proptest` | Retour Phase 0 |
| 4 — Quantification | 2 sem | **G4 : perplexité reproduite** | **Point de sortie** |
| 5 — Noyau GPU | 2–3 sem | G5 : débit vs QTIP | Livrer en documentant |
| 6 — Intégration | 2 sem | G6 : bout en bout | — |
| 7 — Portabilité | 2–3 sem | — | Optionnel |

**Total : 11–15 semaines.** Deux vrais points de sortie, G0 et G4, tous deux **avant** le
gros de l'investissement. C'est délibéré : au terme de la Phase 4, soit on sait que la
méthode tient et il ne reste que de l'ingénierie à risque maîtrisé, soit on a dépensé cinq
semaines au lieu de quinze.

---

## 5. Risques

| Risque | Probabilité | Parade |
|---|---|---|
| **Qualcomm publie son code** | Moyenne | Quatre mois de silence rendent ça peu probable. Et l'implémentation Rust + intégration moteur garde sa valeur même si un portage PyTorch sort. |
| Papier sous-spécifié sur l'indexage | **Élevée** | Gate G0. Contact auteurs immédiat. |
| Chiffres non reproductibles | Moyenne | Gate G4, avant l'investissement noyau. |
| Noyau trop lent | Moyenne | Gate G5. Repli acceptable : gagner en mémoire en perdant un peu en débit. |
| Collision d'indexage silencieuse | Faible mais **critique** | Gate G3 par `proptest`. Ne jamais relâcher ce test. |
| Concurrence sur le créneau | Faible | Surveiller `dmnunez1993/llvq-paper-reproduction` (dormant depuis le 2 juin 2026). |

---

## 6. Première action

1. Ouvrir [arXiv:2603.11021](https://arxiv.org/abs/2603.11021) depuis un poste non filtré.
2. Répondre aux sept points de la Phase 0.
3. Vérifier le calcul de coin de table de la Phase 3 contre la spec réelle.
4. Confirmer les chiffres du tableau comparatif LLVQ vs QuIP# / QTIP / PVQ.

Tant que le point 4 n'est pas confirmé, tout ce document repose sur des résumés de moteurs
de recherche, pas sur le papier.
