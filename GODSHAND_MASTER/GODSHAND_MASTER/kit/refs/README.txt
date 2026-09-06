REFERENCE PHOTOS
================
Two ways to provide photos. Easiest first.

OPTION A - DUMP & AUTO-SELECT (recommended)
  Drop ALL your photos into the _inbox/ folder for each person:
    refs/kunal/_inbox/   <- all of your photos (youth + 40s, any count)
    refs/gunja/_inbox/   <- all ~99 of her photos
  The preflight culler (src/cull_photos.py) then:
    - face-detects + de-duplicates (perceptual hash)
    - quality-scores (sharpness/resolution/exposure/face size)
    - bins by angle (front / three-quarter / profile; body vs close-up)
    - sorts expression (neutral vs smiling; honors the serious-tone rule)
    - auto-picks the top few per bin into face/ body/ costume/
    - writes selection_report.csv  (review & override anything)
  From ~99 it typically keeps ~10-14 of the sharpest, best-angled, well-lit shots.

OPTION B - HAND-PLACE
  Put photos directly into these folders (counts are targets):
    face/     4-6 close-ups (front x2, three-quarter L+R, 1 profile)   min 3
    body/     4-5 full-body (front x2, 3/4, side)                      min 2
    costume/outfit_1/   1-2 full-body shots of a specific outfit
  Per person: ~9-13 ideal, ~5-6 minimum.

QUALITY RULES
  High-res (face >=1024 px, sharp), even soft light, plain background,
  face unobstructed on front shots, recent & consistent appearance,
  at least one neutral expression. Accepted: .jpg .jpeg .png .webp

AGE SPLIT FOR KUNAL (few youth photos, many 40s)
  THE MOMENT is set almost entirely in the early-20s Bangalore era, so young-Kunal
  is used in nearly every shot. Put your few youth photos anywhere in refs/kunal/
  (or refs/kunal/_inbox/); the de-age pass (src/deage.py) builds a consistent
  "young Kunal" reference set into refs/kunal_young/ and uses your 40s photos for
  build/wardrobe. refs/kunal_40s/ is used only for the closing present-day shot (m08).
  Even 3-5 clear youth photos are enough to lock a believable younger face.

ADD A NEW CHARACTER
  Copy the whole _TEMPLATE_person/ folder, rename it, and fill it the same way.
