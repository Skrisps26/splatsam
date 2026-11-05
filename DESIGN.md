---
name: Splatsam
description: Offline synthetic-run replay for inspectable robotics evidence.
colors:
  bg: "#080d18"
  panel: "#101827"
  foreground: "#f1f5ff"
  muted: "#a6b2c7"
  border: "#273247"
  accent: "#9b93ff"
  mint: "#68e6c4"
  error: "#ff8e9d"
  warning: "#f6cb80"
  halo: "#17203c"
typography:
  display:
    fontFamily: "Manrope, sans-serif"
    fontSize: "clamp(3.2rem, 7.8vw, 6.3rem)"
    fontWeight: 800
    lineHeight: 1.05
    letterSpacing: "-0.04em"
  body:
    fontFamily: "Manrope, sans-serif"
  heading:
    fontFamily: "Manrope, sans-serif"
    fontWeight: 700
    letterSpacing: "-0.03em"
rounded:
  small: "10px"
  media: "12px"
  panel: "16px"
  pill: "40px"
spacing:
  compact: "0.8rem"
  panel: "1.4rem"
  page-x: "3rem"
components:
  button-primary:
    backgroundColor: "{colors.accent}"
    textColor: "{colors.foreground}"
    rounded: "{rounded.small}"
    height: "44px"
  panel:
    backgroundColor: "{colors.panel}"
    textColor: "{colors.foreground}"
    rounded: "{rounded.panel}"
    padding: "{spacing.panel}"
---

# Design System: Splatsam

## Overview

**Creative North Star: "The Recorded Run Console"**

Splatsam is a dark, cinematic product-demo replay—not a light evidence notebook. The page begins with an oversized gradient wordmark and a compact REPLAY/provenance bar, then makes one recorded synthetic experiment progressively inspectable. Technical density rises only as saved events arrive: stage progression, media, tuning, and the sealed result.

The visual authority comes from `dashboard/styles.css`, the Streamlit panels, and the observed captures: `docs/screenshots/03-stepper.png` through `07-final.png` show the current desktop sequence; `docs/screenshots/09-mobile.png` shows the current narrow layout. Review captures also exist at `.impeccable/review/desktop.png` and `mobile.png`. In the observed 390px viewport, document and main scroll width were both 390px; reduced-motion media behavior was also observed.

**Key Characteristics:**
- Midnight depth with a restrained top-right blue halo, lilac action color, and mint confirmation color.
- High-contrast headline gradient and compact evidence labels instead of decorative chrome.
- Translucent, bordered panels frame saved media and data without competing with it.
- Motion signals recorded replay state; `prefers-reduced-motion` removes CSS animation and transition.

## Colors

The palette separates recorded-system context from outcomes: cool lilac directs interaction, mint marks completed or final evidence, and rose/warm amber identify rejection or pending states.

### Primary
- **Replay Lilac:** Used for primary actions, active stage treatment, links, labels, and the display gradient.

### Secondary
- **Evidence Mint:** Used for completed stages, success badges, focus outlines, and final metric emphasis.

### Tertiary
- **Failure Rose and Pending Amber:** Used only for rejected, failed, or waiting states.

### Neutral
- **Midnight Field:** The app background and scrollbar track.
- **Layered Navy Panels:** Cards, metrics, code-like traces, media surrounds, and the sidebar.
- **Cool Foreground and Muted Text:** Evidence hierarchy, captions, and labels.
- **Hairline Border:** Defines panels, cards, dividers, and the stepper without a heavy grid.

**The Provenance Contrast Rule.** Do not hide synthetic/replay status in low-contrast decoration; the mode pill, labels, captions, and final-result disclaimer must remain legible against the midnight field.

## Typography

**Display Font:** Bundled Manrope (with `sans-serif` fallback)  
**Body Font:** Bundled Manrope (with `sans-serif` fallback)  
**Label/Mono Font:** UI monospace for parameter diffs and trajectory rows.

**Character:** Dense but not bureaucratic. Tight, heavy display type establishes the demo moment; tabular metrics, compact captions, and monospace traces keep saved evidence scannable.

### Hierarchy
- **Display:** Heavy gradient SplatSAM title; it is large on desktop and constrained at narrow widths.
- **Headline:** Bold, tight section titles for story beats and evidence groups.
- **Body:** Larger lead statement on the landing state; compact explanatory copy elsewhere.
- **Label:** Small, weighted pills, stage labels, provenance captions, and model activity rows.

**The Evidence Hierarchy Rule.** Use display type for the product/replay story, not for outcomes; counts, intervals, and provenance retain a quieter, legible hierarchy.

## Layout

The main container is wide but bounded, with roomy desktop side padding and an intentional first viewport: brand/provenance, statement, saved scene preview, then local upload/sample choice. Replay controls sit above the seven-stage stepper and chapter pills; current evidence fills the main canvas, with tabs or expanders exposing secondary detail.

Desktop evidence uses paired columns for media and context, tuning and charts, or baseline/final clips. At `800px` and below, page padding tightens, the stepper becomes four columns, metrics compress, and the grid-based episode wall narrows from 25 to 20 cells. The observed 390px capture has no horizontal page overflow; media and controls remain within the viewport.

## Elevation & Depth

Depth is tonal rather than shadow-led: a radial background halo, layered navy surfaces, translucent gradient glass panels, thin cool borders, and occasional backdrop blur make evidence feel contained. Media and Plotly charts have gently rounded clipping rather than heavy elevation. The final result card may use a stronger navy-to-teal surface to distinguish the one sealed result.

**The Quiet Surface Rule.** Let the saved media, not box shadow, carry the visual weight. Borders and tonal differences should remain subtle.

## Shapes

Panels use gentle 16px rounding; controls use a more compact 10px rounding; media frames use 12px. Status/mode badges are fully pill-shaped. The stepper is intentionally linear—its colored lower border and circular number/check communicate recorded stage progression more clearly than an ornamental card grid.

## Components

### Buttons
- **Shape:** Compact rounded rectangle with a minimum 44px height.
- **Primary:** Lilac gradient action for starting/restarting the recorded replay.
- **Secondary:** Dark bordered controls for pause/resume and the local sample/upload flow.
- **Disabled:** Reduced opacity makes unavailable live launch clear without changing its meaning.
- **Focus:** Mint 3px outline with offset; do not replace keyboard focus with a color-only state.

### Cards / Containers
- **Corner Style:** Gently curved panels; glass containers reserve 16px rounding.
- **Background:** Layered navy or translucent navy gradient with a hairline border.
- **Internal Padding:** Evidence panels use the panel spacing token; metric cards use tighter padding.
- **Signature use:** The final sealed-result card, upload well, metrics, trace panel, and empty state use the same dark-surface family.

### Inputs / Fields
- **Upload:** Dark dashed, rounded drop zone with muted support copy; uploaded MP4s remain local.
- **Replay controls:** Native buttons, select box, pills, progress, and slider are visually skinned but retain Streamlit behavior.
- **Comparison:** A minimal same-origin iframe contains a labeled range input; it is keyboard-operable and states that differently framed views are not pixel registration.

### Navigation
- **Top bar:** Small product name, explicit “Replaying a recorded run” label, run identity/cost, and elapsed presentation time.
- **Chapters:** Native pill control for Masks, Scale, Physics, Tuning, and Result.
- **Detail disclosure:** Native tabs during progression and expanders after the final reveal preserve the story without hiding data permanently.

### Stepper and Evidence States
- **Stage stepper:** Seven scheduled stages; done, running, pending, and absent status are differentiated by border, dot, and readable text.
- **Status badges:** Mint success, rose rejection, and amber pending states reinforce labels rather than substitute for them.
- **Media:** Saved masks reveal progressively; videos and images use the media radius. An idle orbital mark is only a waiting state, never fabricated evidence.

## Do's and Don'ts

### Do:
- **Do** lead the replay with its recorded/synthetic provenance and keep presentation pacing distinct from runtime.
- **Do** use the lilac-to-mint display gradient sparingly: the brand title and controlled high-value moments only.
- **Do** keep focus outlines, captions, labels, and status text visible in the dark palette.
- **Do** preserve the mobile reflow and reduced-motion override when extending a replay panel.

### Don't:
- **Don't** revert to the obsolete light evidence-notebook palette or a generic white Streamlit surface.
- **Don't** make the replay look like a live job: unavailable Live launch remains visibly unavailable, not hidden behind optimistic copy.
- **Don't** use presentation animation to imply measured pipeline latency or new computation.
- **Don't** present source/simulation comparisons as calibrated pixel alignment or illustrative training clips as evaluation evidence.
