# Product

<!-- impeccable:product-schema 1 -->

## Platform

web

## Users

The study owner rates blind sittings on a desktop, using the keyboard or buttons while inspecting images.

## Product Purpose

Make explicit ratings quick to enter, verify, correct, and resume while preserving the study's blind protocol and the existing sheet workflow.

## Operating Context

The app runs locally against an existing sitting. Items remain in sheet order; sheet creation and collection happen outside the app.

## Capabilities and Constraints

- Keep item identities, source paths, hidden metadata, and rating distributions out of the interface.
- Show the state of each save and prevent advancing when an image is unreadable or required choices are unsaved.
- Preserve existing ratings and support recovery after interruption. Finishing makes a sitting read-only.
- Use synthetic images and ratings when developing or reviewing the interface.

## Accessibility & Inclusion

Support keyboard operation, visible focus, readable controls, and clear save and error feedback.

## Document ownership

This file owns product purpose, users and protocol constraints. `static/style.css` is the current UI implementation; [DESIGN.md](DESIGN.md) is its CSS-derived visual snapshot. [README.md](README.md) owns launch, operation and owner-viewer provenance. [DESIGN-REVIEW.md](DESIGN-REVIEW.md) owns synthetic review evidence and detector limits.
