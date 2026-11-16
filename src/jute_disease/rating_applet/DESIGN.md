---
name: Local blind rating instrument
description: A quiet, focused interface for explicit blind ratings.
colors:
  text: "#222"
  page: "#f4f3f1"
  image-well: "#777"
  muted-text: "#555"
  surface: "#fff"
  selected-choice: "#444"
  divider: "#c9c7c4"
  focus: "#1e4c79"
typography:
  body:
    fontFamily: "system-ui, sans-serif"
    lineHeight: 1.45
rounded:
  sm: ".25rem"
components:
  rating-button:
    backgroundColor: "{colors.surface}"
    textColor: "{colors.text}"
    rounded: "{rounded.sm}"
    padding: ".65rem .8rem"
    typography: "{typography.body}"
  rating-button-selected:
    backgroundColor: "{colors.selected-choice}"
    textColor: "{colors.surface}"
    rounded: "{rounded.sm}"
    padding: ".65rem .8rem"
    typography: "{typography.body}"
---

# Design System: Local blind rating instrument

## Overview

**Creative North Star: “The Quiet Instrument”**

This is an operate interface for one reviewer working through an existing blind sitting. The observed visual system gives the images and explicit choices most of the attention. Neutral page and control colors, system typography, and simple borders support scanning without adding visual cues about an answer.

The values below are a snapshot extracted from `static/style.css`. That stylesheet remains the implementation source of truth. The interface currently has no shadow or motion rules.

**Key Characteristics:**
- Neutral page and control surfaces with a gray image well.
- System UI typography and compact heading hierarchy.
- Equal-sized pair panels with synchronized scrolling.
- Visible keyboard focus and selected-choice state.

## Colors

The stylesheet uses neutral surfaces and text, with a blue focus outline as its only saturated accent.

### Primary
- **Focus blue** (`#1e4c79`): Visible keyboard-focus outline.

### Neutral
- **Text** (`#222`): Default foreground inherited by controls.
- **Page** (`#f4f3f1`): Root background.
- **Image well** (`#777`): Background behind images.
- **Muted text** (`#555`): Shortcut labels on unselected choices; selected shortcuts use white.
- **White surface** (`#fff`): Default button background and selected-choice text.
- **Selected charcoal** (`#444`): Pressed-choice background.
- **Divider** (`#c9c7c4`): Pair-rubric separator.

## Typography

**Body Font:** `system-ui, sans-serif`

**Character:** Native system UI shapes keep the instrument familiar and legible. Buttons inherit the page font.

### Hierarchy
- **Body** (root line-height `1.45`): Default interface text.
- **Title** (`1.5rem`): Main heading.
- **Section heading** (`1.15rem`): Second-level heading.
- **Subheading** (`1rem`): Third-level heading.
- **Emphasis** (weight `600` or `700`): Pair captions, legends, and section labels.

## Layout

The main region is centered up to `1900px` with `1.5rem` padding. Above `1050px`, the workspace gives the image region more width with `1.7fr` images and a `minmax(22rem, 1fr)` choice column separated by `1.5rem`. Pair frames remain equal-width and use `min(56vh, 660px)` height so more of the viewport is available for image inspection. Between `961px` and `1050px`, a compact two-column workspace keeps image panels and rating buttons visible together; below `961px`, choices move beneath the images. Pair panels stack below `640px`. At `1000 × 700`, each pair panel remains at least `260px` high, all four rating buttons fit above the sticky navigation, and the definitions can continue below the viewport. Image wells scroll when scaled content exceeds the frame; pair scrolling is synchronized, and zoom keeps the approximate scroll position when the image bounds change. Sitting navigation stays sticky at the bottom when needed.

## Elevation & Depth

The current stylesheet defines no shadows. It separates content with neutral backgrounds, borders, spacing, and the gray image well.

## Shapes

Buttons use a `0.25rem` corner radius and a thin border. Image panels are rectangular and use clipping through scrollable frames rather than rounded image masks.

## Components

### Rating buttons
- **Default:** White surface, dark foreground, thin dark border, inherited system font, and `.65rem .8rem` padding.
- **Selected:** Charcoal background with white text.
- **Disabled:** Reduced opacity and default cursor.
- **Focus:** `3px` solid blue outline with `3px` offset.

### Image panels
- **Pair layout:** Two equal-width panels at wide sizes, each sized to the available image column and viewport height.
- **Image fit:** Previews and originals preserve aspect ratio. Shared 1×–4× zoom works in both modes; fit preview scales both sides by the same rule and may enlarge a small preview without adding detail. Full-resolution images and enlarged previews can scroll inside the frame.
- **Orientation:** Each image has labeled rotate, flip, and reset controls and a text status. Transformations affect the displayed image only and follow its panel when swapped.
- **Finish flow:** Separate finish and confirmation controls keep accidental repeated activation from confirming a read-only transition. Confirmation is keyboard reachable, Escape cancels it, and a changed rating disarms it.

## Do's and Don'ts

### Do:
- **Do** preserve equal pair-panel sizing and the image aspect ratio.
- **Do** keep keyboard focus visible with the stylesheet's focus outline.
- **Do** keep selected choices visibly distinct from default choices.

### Don't:
- **Don't** add decorative shadows or motion that are absent from the current interface.
- **Don't** add visual styling that suggests a preferred rating.
