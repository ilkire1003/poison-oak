# Poison Oak ID Trainer — Design Doc

Date: 2026-11-19
Status: approved (verbal greenlight, 2026-11-19 — no further design docs planned; future decisions get approved directly in chat)

## Intent and why

Pacific poison oak (*Toxicodendron diversilobum*) is hard to tell apart from
harmless lookalikes, especially as young growth, and getting it wrong has a
real physical cost. Erik's professor framed the pedagogical angle as
"contrasting cases": don't just classify, teach the human eye to see the
difference without a phone in hand. The deliverable is a Kahoot-style quiz
(practice mode with feedback, then a no-feedback 10-question test) that shows
real plant photos and explains, in a short line, what actually gives the
plant away.

This is academic/personal work, professor-advised, with a hard 3-week
deadline.

## Goals

- A working practice mode: 4 images per round (1 poison oak + 3 lookalikes),
  user taps one, sees right/wrong + a short per-image explanation, repeatable.
- A working test mode: 10 rounds, no feedback mid-test, final score out of 10.
- Each explanation is grounded in an actual visual cue in that image, not a
  generic species blurb.
- Runs as a responsive website, installable as a PWA, no account/login.
- Covers both young and mature growth stages of poison oak.

## Non-goals

- No live camera-based "point your phone at a plant" classification.
- No live VLM Q&A in v1 (explicitly deferred — hardcoded explanations only).
- No user accounts, leaderboards, or multi-user features.
- No native/React Native app.
- No claim of medical/safety certification — this is an educational aid, not
  a diagnostic tool.

## Anti-goals (would make this a failure even if "technically working")

- A quiz that's really just "memorize these 40 photos" — explanations must
  reference the visual cue (leaflet shape, leaf margin, sheen, growth habit,
  reddish new growth, "leaves of three", clustering, bark/vine structure
  etc.), not a circular "this is poison oak because it's poison oak."
  Confusable species that are visually the closest get precedence so the
  quiz does its contrasting-cases job instead of being a free-win round.
- A demo that only works on young-growth photos, or only mature — the
  professor's ask was explicitly both stages.
- Shipping with zero explainability layer — Grad-CAM is the floor, not a
  stretch goal (see Risk-managed sequencing).

## Constraints

- **3-week deadline**, effectively solo build.
- Dataset images must be properly licensed / usable in a public demo →
  iNaturalist/GBIF, not generic scraping.
- **No local GPU.** All training (baseline classifier, InfoNCE pretraining,
  and any diffusion work) runs on HPC, which has A100/H100 access. This
  means HPC access is a day-one dependency, not a week-3 fallback — queue
  times and job setup need to be sorted before Week 1's baseline training,
  not discovered when something's already late.
- No budget assumed for paid APIs/compute beyond what's already available
  to Erik.
- No professor review/sign-off gate — the professor said Erik runs the
  project and isn't a plant specialist himself, so species-list and
  explanation-text accuracy is Erik's call to make and verify (e.g. against
  regional plant guides), not a scheduled review step.

## Approaches considered

### Explanation mechanism
- **Live VLM Q&A** — flexible, answers follow-up questions, but adds a
  serving dependency, cost, and failure modes (hallucinated plant facts) —
  risky with a 3-week clock and zero room for a flaky demo.
- **Hardcoded per-image explanations (chosen)** — written once during
  content prep, reviewed for accuracy, zero runtime risk. VLM stays a
  stretch goal only if free-form Q&A becomes a hard requirement later.

### Explainability / "why" layer
- **LIME/SHAP** — Erik's original idea; works on arbitrary models but is
  noisy on images and slow to generate convincing output.
- **Grad-CAM only** — cheap, fast, standard for CNN image classifiers,
  produces a heatmap over the leaf. Good floor.
- **Grad-CAM + diffusion counterfactual edit (chosen)** — Grad-CAM shows
  *where* the model is looking; a diffusion-based counterfactual nudges the
  image's embedding across the decision boundary and shows *what minimal
  visual change* would flip the call (e.g. "if the leaflet edge were smooth
  instead of lobed, this would look like poison oak"). This is the most
  direct realization of "contrasting cases" and of the professor's diffusion
  suggestion, but it's also the most novel/least certain-to-work piece on a
  3-week clock (see Risk-managed sequencing below for how this is protected
  against).

### Model architecture
- **Single end-to-end classifier** — simplest, fastest to get a baseline
  working, but gives Grad-CAM a weaker "why" signal and doesn't naturally
  support the diffusion counterfactual (which wants an embedding space to
  push the image across).
- **Two-stage: InfoNCE contrastive pretraining → embedding space → light
  classifier head (chosen)** — pretrain embeddings with InfoNCE using the
  most visually-confusable lookalike per species as hard negatives (this is
  the literal "contrasting cases" the professor asked for, applied to
  training, not just quiz design). Grad-CAM and the diffusion counterfactual
  editor both read from the same embedding space, so the "why" stays
  consistent across both explanation types.

### Dataset sourcing
- **Generic web scraping (Erik's original idea)** — fast but unreliable
  labels, unclear licensing for a public demo.
- **iNaturalist/GBIF (chosen)** — geotagged, expert-verified-ish,
  machine-readable licenses, good species coverage for lookalikes, directly
  queryable by API. Images get resized/scaled during prep as needed.
  License handling (my call): prefer CC0, accept CC-BY, and show a visible
  per-image credit line in the quiz UI for any CC-BY image used — avoids a
  late scramble if CC0-only coverage turns out too thin.

### App delivery
- **Native or React Native app** — better phone feel, much higher build
  cost for a 3-week solo build.
- **Responsive website, installable as PWA (chosen)** — this is a demo, not
  a shipped product; one codebase, no app-store friction, works on the
  professor's phone or laptop immediately. Confirmed by Erik given the
  deadline.

### App architecture
- **Live model inference in the app** — unnecessary: the quiz only ever
  shows curated images from the prepared deck, never a user's own camera
  photo, so there's nothing to classify live.
- **Static precomputed content, app is a pure quiz player (chosen)** — all
  training, Grad-CAM generation, and diffusion counterfactual generation
  happen once, offline, during content prep. The shipped app loads a JSON
  deck (image + correct answer + short explanation text + optional overlay
  image) and plays the quiz. Zero backend, zero inference latency, zero
  "model doesn't load on this phone" risk.

## Decided (the how, as already settled)

- Dataset: pull poison oak + confusable lookalike species from
  iNaturalist/GBIF, covering young and mature growth stages, filtered to
  usable licenses, resized for training and for display.
- Explanations: short, hardcoded, one per image, written to name the actual
  visual cue in that photo.
- Model: InfoNCE-pretrained embedding space (hard negatives = closest
  lookalike per species) → lightweight classifier head on top.
- Explainability: Grad-CAM (floor) + diffusion counterfactual edit (target),
  both derived from the InfoNCE embedding space.
- App: single responsive website, installable as PWA, zero live inference —
  reads a precomputed quiz-content deck.
- Modes: practice (repeatable, 4-choice, feedback + explanation per
  question) and test (10 questions, no feedback until a final score).

## Risk-managed sequencing (my call, given the 3-week deadline)

The diffusion counterfactual piece is the most technically novel part of
this project and the least certain to land cleanly in 3 weeks. To guarantee
there's a working, gradeable demo no matter what, the build goes in this
order, with each stage independently demoable. Clock assumed to start today,
2026-11-19, ending ~2026-12-10 — flag if the 3 weeks are already partway
through.

1. **Week 1, day 1** — two things in parallel, before anything else: (a) pull
   real iNaturalist/GBIF counts for poison oak and each candidate lookalike,
   split by young vs. mature growth stage and by usable license — if
   coverage for a species/stage is too thin for contrastive hard negatives,
   swap that lookalike or fall back to a simpler classifier for that
   pairing, found out immediately, not after the baseline model is already
   built; (b) get HPC access/queue time confirmed and a test job running,
   since there's no local GPU and everything from the baseline classifier
   onward depends on it.
1b. **Week 1** — dataset collection + cleaning, baseline classifier trained,
   Grad-CAM wired up. This alone is already a defensible demo of "contrasting
   cases + explainability."
2. **Week 2** — swap in InfoNCE pretraining + classifier head on the
   embeddings (replaces the baseline classifier); in parallel, build the
   quiz app (static site/PWA, practice + test modes) against placeholder
   content so app work isn't blocked on the model.
3. **Week 3** — attempt the diffusion counterfactual explanation layer.
   If it works, it ships alongside Grad-CAM. If it's not converging or
   producing convincing edits by roughly the midpoint of week 3, cut it and
   ship with Grad-CAM-only explanations — the quiz and written explanations
   don't depend on it, so this is a clean fallback, not a scramble.

Flag if this ordering doesn't match how you want to risk-manage the
deadline — it's my call to make, not a decision you've already stated.

## What's left to the builder

- Exact list of lookalike species to include (and which is the "hard
  negative" pairing for each poison-oak growth stage) — needs domain
  research, likely from the professor or regional plant guides.
- Exact CNN/ViT backbone choice for the classifier head.
- Exact diffusion model/method for the counterfactual edit (e.g. a guided
  diffusion edit conditioned on crossing the embedding decision boundary vs.
  a simpler SDEdit-style approach) — this is explicitly left open pending
  week 3's time budget.
- Number of quiz images needed per species/growth-stage to get reasonable
  coverage without over-scoping dataset collection.
- Visual design of the quiz app (not yet discussed).
- Where the project's code repo lives (no repo exists yet; this doc lives in
  the OpenClaw workspace for now and should move into the project repo once
  created as an early implementation step).
