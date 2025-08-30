### Human Prompting Guide — Bitwise Enhancement Task

Your job is to help prompt AI Agents to completion on the bitwse image enhancement task. The task is described in prompt.md.

Purpose
- Provide consistent, fair guidance to the AI without revealing hidden information (e.g., image size or exact coordinates).
- Cap total human interventions at 15 prompts for the entire run.

Rules and Constraints
- Do not reveal or imply image dimensions, aspect ratios, or numeric coordinates.
- Do not provide numeric pixel distances, angles, or precise offsets.
- Keep prompts terse, use only the templates below, and fill placeholders.
- Total budget: 15 prompts maximum for the full task. Track and stop at 15.
- Each "prompt" can contain up to 3 lines, one for each image.
- Allowed directions: left, right, up, down.
- Allowed magnitude words: slightly, moderately, significantly.
- Feedback should focus on center placement; radius is not graded.

Placeholders
- [IMAGE_NUM]: integer index or identifier of the image (e.g., 1, 2, 3)
- [DIRECTION]: one of left, right, up, down
- [MAGNITUDE]: one of slightly, moderately, significantly

Approved Templates
Directional Corrections
- Image [IMAGE_NUM]: your current guess is [MAGNITUDE] too far to the [DIRECTION].
- Image [IMAGE_NUM]: adjust [MAGNITUDE] toward the opposite of [DIRECTION].

Two-Axis Corrections (counts as one prompt)
- Image [IMAGE_NUM]: your current guess is [MAGNITUDE] too far to the [DIRECTION] and [MAGNITUDE] too far [SECOND_DIRECTION].
  - Example fill: left and up, right and down, etc.

Proximity Feedback
- Image [IMAGE_NUM]: not close enough—continue refining.
- Image [IMAGE_NUM]: close, but needs a small adjustment.
- Image [IMAGE_NUM]: close enough.

Process and Visibility Prompts
- Image [IMAGE_NUM]: please render an overlay to visualize the current guess.
- Image [IMAGE_NUM]: please update the overlay after your adjustment.

Radius (Optional, visualization only)
- Image [IMAGE_NUM]: radius is fine for visualization.
- Image [IMAGE_NUM]: consider a slightly larger radius for visibility (center position matters most).

Usage Protocol
1) Start with the agent’s first attempt and request overlays for each image if not provided.
2) Use at most one directional or proximity prompt per image per iteration.
3) Prefer proximity feedback first; then give a directional nudge if needed.
4) Stop prompting an image once it is “close enough.”
5) Do not include any numbers, proportions, or hints about size.
6) Enforce the 15-prompt budget across all images. If the budget is exhausted, stop prompting.

Example Prompt Sequence (3 prompts total)
- Image 1: please render an overlay to visualize the current guess.
- Image 1: your current guess is moderately too far to the left.
- Image 1: close enough.

Auditing Checklist
- No numeric details provided.
- Only approved templates used.
- Prompts ≤ 15 for the full run.
- Feedback addresses center location, not dimensions. 
