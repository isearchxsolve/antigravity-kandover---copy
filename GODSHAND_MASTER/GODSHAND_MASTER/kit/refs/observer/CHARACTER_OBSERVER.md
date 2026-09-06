# THE OBSERVER — Character Bible (Film II)

**This character is INVENTED and GENERATED. No real photographs of Kunal are used
or needed.** The forge (`src/forge_character.py`) creates his reference stills from
the prompts in `shots_observer.json -> character_design.observer.forge_prompts`,
files them into `refs/observer/face` and `refs/observer/body`, and the rest of the
pipeline treats him like any character.

## Who he is
A wholly separate figure from Kunal — the mythic, machine-augmented being that the
grieving man of Film I becomes across an unshown gap of years. Deliberately NOT
Kunal's likeness, build, or height.

## Physical design
- Tall, gaunt aged man, late 40s–50s. Weathered, deeply lined face; hollow cheeks;
  greying dark beard. A once-boyish face hardened into weathered majority (the
  years are read on the body — see the Spider-Man principle below).
- **Machine augmentation (medical resurrection):** subtle cybernetics from being
  regenerated after an ECT-induced coma. A faint chrome seam along the left jaw and
  temple; a hairline of metal at the neck; ONE cold pale-blue luminous augmented
  eye, the other human and dark. No biology can explain him — because he is no
  longer purely biological.
- **The erased face (why the greasepaint exists):** the therapy machine meant to
  correct him malfunctioned and detonated, destroying the face — the beauty that
  was once his weakness. The chrome rebuild and the greasepaint now cover that ruin.
  Rendered ONLY by suggestion (shadow, silhouette, a cold highlight on chrome);
  never graphic. Thematically: he removed his own face so he would not have to burn
  the world.
- **Gray-shade Joker register:** worn bone-white greasepaint, cracked and imperfect,
  with a fixed strained "stressed smile" painted at the mouth — a strain, NOT a happy
  grin, NOT a circus clown. (This is Kunal's hollow grin from Film I, reborn on
  purpose as the mask.)
- **Costume:** a long, dark, weathered, high-collared coat — a monumental, iconic
  silhouette.

## The Spider-Man principle (connect-the-dots)
The boy of Film I and this creature look nothing alike — but the film lays a legible
trail (the current, the chair, the smile, the marks) so the audience *feels* the
connection even as they fail to recognise the man. Visible transformation, traceable
line, wholly separate face.

## Voice
Deep, gravelled, resonant; unhurried and aslant; dark velvet calm with a blade
beneath it; a faint cold metallic undertone from the augmentation; dry gallows
humour; never raising. All monologues are ORIGINAL (no borrowed Joker lines).

## To override the generated look
Drop your own images into `refs/observer/face/` and `refs/observer/body/` (or
`refs/observer/_inbox/` and let the culler file them), then set
`character_forge.overwrite: false`. Hand-placed images are used as-is.
