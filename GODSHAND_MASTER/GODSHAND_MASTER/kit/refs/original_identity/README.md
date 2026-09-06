# original_identity/  -- your OWNED synthetic character

Drop 5-15 stills (and optionally a short reference video) of an ORIGINAL, synthetic
character you own the rights to. Front + 3/4 + profile angles, neutral lighting, at
least 1024px. Consistent face/hair/wardrobe so the identity holds across shots.

Then in config.yaml set:

    identity_override:
      enabled: true
      refs_dir: refs/original_identity

Every character in every shot will then use THIS identity instead of the real-actor
references (e.g. hero_alam_raveena.mp4). This is the personality-rights fix for
monetized clips in India: do NOT sell clips built on a real person's likeness.

Tip: generate this identity once (a few seed images from a text-to-image model, or a
single Wan2.2 still), lock the seed, and reuse it as your recurring on-screen talent.
