#!/usr/bin/env python3
"""build_lastlaugh.py -- assemble THE LAST LAUGH (full ~8-9 min Heath Ledger tribute).

ActI (THE GIFT) + ActII (THE MASK), authored here, are chained AHEAD of the
existing 26-beat confrontation (shots_confrontation.json) as the film's climax.

Dignity law (see the Notion treatment): the protagonist Jack is an original
Ledger-order figure, never a scraped replica; the descent is immersion,
insomnia and isolation -- NEVER medication, never a restaging of a real death;
the Joker's fictional end stays the character's. 'Living the character he
performed' is the spine of Act II.

Writes: shots_lastlaugh.json, script/lines_lastlaugh.csv,
audio/prompts_lastlaugh.csv, config_lastlaugh.yaml  (all validated after).
"""
import os, json, csv, copy
import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

TONE = ("cinematic realism, shot on 35mm with fine film grain, subtle anamorphic flare; "
        "deliberate disciplined composition with generous negative space, the figure often small and isolated; "
        "a single motivated light source, soft chiaroscuro, deep quiet shadows; shallow depth of field, slow patient camera; "
        "unhurried contemplative pacing; serious, restrained, melancholic, reverent, intimate; NOT a happy ending")

NEG = ("nudity, nude, skin exposure, cleavage, bare shoulders, revealing or short clothing, sensual, seductive, suggestive, "
       "vulgar, sexual contact, kissing, extra fingers, deformed hands, blurry, low quality, shouting, exaggerated acting, "
       "soap opera, comedy, slapstick, circus clown makeup, rainbow colors, bright flat daylight, sitcom lighting")

VD_JACK = ("deep male voice like a wounded lion, the baritone worn and wistful, a haunted memoir tone, a distant ache under the calm, "
           "breath catching softly at the wound, unhurried, never shouting")
VD_CHOR = ("distant reverent whispered chorus, low doubled murmur, almost subliminal, cavernous reverb, solemn and mythic")
VD_JULIE = ("regal female voice, glacial composure with a fragile warmth beneath, luminous and formidable, restrained, close-mic, never theatrical")

def vd(sp):
    return {"jack": VD_JACK, "julie": VD_JULIE, "chorus": VD_CHOR}[sp]

def b(bid, mov, cam, pal, dur, prompt, lines, music, amb, chars=None):
    if chars is None:
        chars = ["jack"]
    return dict(id=bid, mov=mov, cam=cam, pal=pal, dur=dur, prompt=prompt,
                lines=lines, music=music, amb=amb, chars=chars)

BEATS = [
 # ===================== ACT I : THE GIFT =====================
 b("gift_child", "the_gift", "wide, static, a small figure in a doorway",
   "cold morning light, pale and grey, film grain", 8,
   "a slight young boy alone in a doorway watching a world that never quite includes him, feeling everything too deeply, no face paint, gentle and luminous, monumental negative space, fully modest, cinematic realism",
   [("jack","I felt everything too much, too young. The world has no shelf for a boy like that."),("chorus","Some are born with the volume turned all the way up.")],
   "original tender aching origin theme, a lone felt piano over soft sustained strings, fragile and luminous, wholly original and transformative, instrumental only, no vocals, no lyrics",
   "a quiet morning room tone, a distant street, a child's held breath"),
 b("gift_unseen", "the_gift", "wide, static, figure small and distant",
   "cold desaturated grey-blue, a single pale window light, film grain", 8,
   "a thin quiet young man moving through a bright indifferent city, no one turning to look at him, an ordinary unseen man before any mask, no face paint, monumental negative space, fully modest, cinematic realism",
   [("jack","Before the gift, before the paint, I was only a thin quiet man no one turned to look at."),("chorus","The world looks away first. It always does.")],
   "original quiet aching memoir theme, felt piano and distant cello over soft strings, lonely and searching, wholly original and transformative, instrumental only, no vocals, no lyrics",
   "a bright indifferent city, distant crowd, a lonely wind"),
 b("gift_stagedoor", "the_gift", "static, a man at the edge of a pool of light",
   "warm stage light spilling into cold dark, film grain", 8,
   "a young man lingering at the edge of a theatre stage door, hungry for a lit world that has never let him in, no face paint, longing on his face, single warm light against deep shadow, fully modest, cinematic realism",
   [("jack","I stood at the edge of the light for years, certain it was meant for someone else.")],
   "original memoir theme, piano and cello circling with a fragile hope, wholly original and transformative, instrumental only, no vocals, no lyrics",
   "muffled applause behind a door, a quiet corridor, a held breath"),
 b("gift_outsider", "the_gift", "static wide, a figure outside a lit window",
   "cold blue night against warm window glow, film grain", 8,
   "a young man outside a bright window at night watching people laugh together inside a warmth he is not part of, no face paint, longing and separateness, cold blue exterior against warm interior glow, fully modest, cinematic realism",
   [("jack","I was always on the wrong side of the glass, watching the warmth and learning it by heart.")],
   "original memoir theme, a lone piano over cold strings, wistful separateness, wholly original and transformative, instrumental only, no vocals, no lyrics",
   "a cold street at night, muffled laughter from inside, a lonely wind"),
 b("gift_hunger", "the_gift", "slow push-in, close on hands and eyes",
   "single bare bulb, warm against black, film grain", 8,
   "a young man alone in a bare rented room mouthing lines to no audience, rehearsing a life he has not been given, no face paint, quiet obsession, single bare bulb, deep shadow, fully modest, cinematic realism",
   [("jack","So I practised in the dark, for no one, night after night, as if wanting it enough could make it true.")],
   "original memoir theme, a determined piano figure rising under soft strings, wholly original and transformative, instrumental only, no vocals, no lyrics",
   "a bare room tone, a page turning, a soft steady breath"),
 b("gift_audition", "the_gift", "static wide, a man alone before an unseen table",
   "cold overhead light, grey, film grain", 8,
   "a young man standing alone in an audition room before an unseen panel, trembling and luminous, no face paint, everything at stake in a single moment, cold overhead light, monumental negative space, fully modest, cinematic realism",
   [("jack","Then one door opened. One room. One chance to show them what I carried.")],
   "original memoir theme, suspended strings and a held piano note, fragile anticipation, wholly original and transformative, instrumental only, no vocals, no lyrics",
   "a cold quiet room, a clock, a nervous breath"),
 b("gift_stillroom", "the_gift", "slow push-in on the face",
   "a single hard key light warming, film grain", 8,
   "the same young man mid-performance, something incandescent breaking out of the small quiet man, the room going utterly still around him, no face paint, transfixing, single hard key light, deep shadow, fully modest, cinematic realism",
   [("jack","And when I finally spoke, the room went still. For the first time, the world stopped looking away."),("chorus","The gift was always there. It only needed a door.")],
   "original memoir theme swelling, piano and strings blooming into first warmth, wholly original and transformative, instrumental only, no vocals, no lyrics",
   "a held silence, then a single caught breath from the dark"),
 b("gift_firstlight", "the_gift", "crane up, a figure lit on a stage",
   "warm gold stage light against deep black, film grain", 8,
   "a young man alone in a shaft of gold stage light for the first time, the dark auditorium beyond, luminous and alive, no face paint, a gift catching fire, single motivated light, monumental negative space, fully modest, cinematic realism",
   [("jack","They gave me the light. And the light, it turned out, loved me back.")],
   "original ascent theme, warm strings and rising piano, a luminous swell of arrival, wholly original and transformative, instrumental only, no vocals, no lyrics",
   "a vast auditorium, distant swelling applause, a stage hum"),
 b("gift_theonewhosaw", "the_gift", "static two-shot across a table",
   "warm lamp light against deep shadow, film grain", 8,
   "a quiet room: a luminous regal woman actually looking at the young man and seeing his mind for the first time, across a plain table, he half-turned barely daring to meet her gaze, no face paint, a fragile spark of being known, warm single lamp light, deep shadow, fully modest, cinematic realism",
   [("jack","And then she looked at me, not at the gift, at the man behind it, and saw a mind worth keeping."),("chorus","He should have called her a woman, not a goddess.")],
   "original memoir theme, tender solo cello over felt piano, a fragile spark of hope, wholly original and transformative, instrumental only, no vocals, no lyrics",
   "quiet interior room tone, a clock, a held breath", chars=["jack","julie"]),
 b("gift_recognition", "the_gift", "slow push-in, close two-shot",
   "soft warm key, deep shadow, film grain", 7,
   "the young man and the luminous woman in quiet conversation, a rare understanding passing between them, no face paint, warmth and restraint, single soft key light, deep shadow, fully modest, cinematic realism",
   [("jack","For a while, someone in the world knew my name and meant it kindly. I did not know how rare that was.")],
   "original memoir theme, warm intertwining strings and piano, a tender lull, wholly original and transformative, instrumental only, no vocals, no lyrics",
   "close room tone, soft rain on a window, two quiet breaths", chars=["jack","julie"]),
 b("gift_worship", "the_gift", "slow push-in, eyes closed",
   "soft warm key fading to cold, deep shadow, film grain", 8,
   "the young man with his eyes shut turning his face from a soft radiant light, worshipping from a distance and never once speaking, no face paint, reverence shadowed by fear, single motivated key light, deep shadow, intimate, fully modest, cinematic realism",
   [("jack","I loved her the only way I knew, from a distance, in silence, certain a goddess would fall if I reached."),("chorus","So he shut his eyes, and called it love.")],
   "original memoir theme, longing piano and cello circling under sustained strings, wholly original and transformative, instrumental only, no vocals, no lyrics",
   "soft room tone, a slow exhale, distant muffled time passing"),
 b("gift_unspoken", "the_gift", "static, two figures apart in a wide frame",
   "cold blue with a single warm lamp, film grain", 7,
   "the man and the woman standing apart in a large quiet room, a word he cannot say hanging in the air between them, no face paint, ache and restraint, generous negative space, single warm lamp against cold, fully modest, cinematic realism",
   [("jack","There were a hundred nights I nearly told her. I let every one of them pass.")],
   "original memoir theme, a suspended violin and soft piano, an unspoken ache, wholly original and transformative, instrumental only, no vocals, no lyrics",
   "a large quiet room, a distant clock, a held breath", chars=["jack","julie"]),
 # ===================== ACT I : THE ASCENT =====================
 b("asc_firstrole", "the_ascent", "montage-style, a man in costume under lights",
   "warm gold and deep black, film grain", 7,
   "the young man transformed by his first real role, alive under the lights, the world beginning to turn toward him, no clown paint, luminous, single hard key, deep shadow, fully modest, cinematic realism",
   [("jack","The roles came. Each one asked me to disappear a little further into someone else.")],
   "original ascent theme, propulsive warm strings and piano building, wholly original and transformative, instrumental only, no vocals, no lyrics",
   "a film set hum, distant voices, a camera shutter"),
 b("asc_thework", "the_ascent", "montage, a man deep in the craft under lights",
   "warm gold and deep black, film grain", 7,
   "the man wholly alive inside the work, transformed take after take, the good years when the gift was pure joy, no clown paint, luminous absorption, single hard key, deep shadow, fully modest, cinematic realism",
   [("jack","For a while it was pure joy, to vanish into someone else and come back with something true.")],
   "original ascent theme, warm strings and a bright piano, joyful momentum, wholly original and transformative, instrumental only, no vocals, no lyrics",
   "a working set hum, distant crew voices, a focused breath"),
 b("asc_rising", "the_ascent", "crane, a figure crossing a bright plaza",
   "cold bright day softened by grade, film grain", 7,
   "the man walking through a world that now turns to watch him, posters and lights blurred behind, no clown paint, ascendant yet strangely alone, monumental scale, fully modest, cinematic realism",
   [("jack","They called it a rise. From inside, it only felt like running.")],
   "original ascent theme, driving strings and low pulse, momentum with a shadow beneath, wholly original and transformative, instrumental only, no vocals, no lyrics",
   "a busy plaza, distant crowd, camera clicks, wind"),
 b("asc_applause", "the_ascent", "reverse from stage into a dark house",
   "hot stage light into black, film grain", 7,
   "the man on a stage facing a roaring dark auditorium, blinded by light, applause crashing over him, no clown paint, triumphant and hollow at once, single blazing key, fully modest, cinematic realism",
   [("jack","The applause was an ocean. And an ocean is a lonely thing to drown in.")],
   "original ascent theme at full swell, brass and strings cresting, wholly original and transformative, instrumental only, no vocals, no lyrics",
   "a roaring auditorium of applause, a stage hum, a single breath"),
 b("asc_crowned", "the_ascent", "static portrait, low angle",
   "gold and deep shadow, film grain", 7,
   "a portrait of the man at his height, luminous and adored, no clown paint, a faint unease behind the eyes, single hard key, deep black background, fully modest, cinematic realism",
   [("jack","At the top I had everything a man could want, and a growing certainty I did not deserve to keep it.")],
   "original ascent theme, a regal brass line over strings, majestic and uneasy, wholly original and transformative, instrumental only, no vocals, no lyrics",
   "a quiet after-show room, distant muffled crowd, a held breath"),
 b("asc_price", "the_ascent", "static, close on eyes that have seen too much",
   "cold light, deep shadow, film grain", 7,
   "close on the man between roles, a first shadow of cost behind the eyes, the gift beginning to ask for more than it gives, no clown paint, quiet unease, single soft key, deep shadow, fully modest, cinematic realism",
   [("jack","But every character I gave a life to seemed to take a little of mine to pay for it.")],
   "original ascent theme cooling, a warm line shadowed by a low cello, wholly original and transformative, instrumental only, no vocals, no lyrics",
   "a quiet room, a faint hum, a slow thoughtful breath"),
 b("asc_lonelycrowd", "the_ascent", "wide, a lone figure amid blurred people",
   "cold blue night, film grain", 7,
   "the man motionless in the middle of a blurred moving crowd at a bright event, utterly alone in the noise, no clown paint, isolation at the peak, generous negative space, fully modest, cinematic realism",
   [("jack","The more they loved the work, the less they could see the man who made it.")],
   "original ascent theme thinning, strings receding into a lone piano, wholly original and transformative, instrumental only, no vocals, no lyrics",
   "a loud party blurred to a hum, distant laughter, a lonely quiet"),
 b("asc_herlight", "the_ascent", "static two-shot, warm pool of light",
   "warm amber against cold, film grain", 7,
   "the man and the luminous woman side by side at the edge of a bright event, the one true thing in a false room, no clown paint, unspoken tenderness, single warm light, deep shadow, fully modest, cinematic realism",
   [("jack","She was the only room I felt real in. And still I never said the word.")],
   "original memoir theme returning warm, cello and piano, a tender fragile lull, wholly original and transformative, instrumental only, no vocals, no lyrics",
   "a distant event hum, two quiet breaths, soft footsteps", chars=["jack","julie"]),
 b("asc_offer", "the_ascent", "static, a script on a table under a lamp",
   "cold light on paper, deep shadow, film grain", 8,
   "the man alone at a table staring at a script under a single lamp, a role unlike any other waiting on the page, no clown paint, fascination edged with dread, single hard light, deep black, fully modest, cinematic realism",
   [("jack","Then the role arrived. The one that would ask for everything, and promise nothing back."),("chorus","Some doors only open one way.")],
   "original theme turning, warm strings cooling as a low unease enters beneath the piano, wholly original and transformative, instrumental only, no vocals, no lyrics",
   "a quiet study, a lamp hum, a page turning, a held breath"),
 b("asc_thescript", "the_ascent", "slow push-in on eyes reading",
   "cold blue, single lamp, film grain", 7,
   "close on the man's eyes moving across the page, something in the character reaching back out of the paper toward him, no clown paint, dawning recognition, single lamp light, deep shadow, fully modest, cinematic realism",
   [("jack","I read him once and knew him like an old wound. That should have been the warning.")],
   "original theme, a low cello and suspended strings, unease circling, wholly original and transformative, instrumental only, no vocals, no lyrics",
   "a quiet room, a page turning slowly, a low drone rising"),
 b("asc_temptation", "the_ascent", "static, a man before a dark mirror",
   "cold light, a dim mirror, film grain", 7,
   "the man standing before a dim mirror, the reflection holding a fraction too long, the character already waiting inside the glass, no clown paint yet, quiet dread, single cold light, deep shadow, fully modest, cinematic realism",
   [("jack","To play him truly, I would have to let him live in me. I told myself I could evict him after.")],
   "original theme, a cold piano figure over a low drone, temptation and dread, wholly original and transformative, instrumental only, no vocals, no lyrics",
   "a quiet room, a faint low hum, a slow breath"),
 b("asc_threshold", "the_ascent", "static wide, a man in a doorway of light and dark",
   "warm light behind, cold dark ahead, film grain", 8,
   "the man standing in a doorway between a warm lit room and a vast dark one, about to step from one into the other, no clown paint, a threshold moment, monumental negative space, fully modest, cinematic realism",
   [("jack","So I said yes. And yes, it turned out, was a door that only opened one way."),("chorus","The man walked in. Something else would walk out.")],
   "original theme darkening, low strings swelling under a final warm piano note, wholly original and transformative, instrumental only, no vocals, no lyrics",
   "a threshold, a warm room behind and a cold vast dark ahead, a single step"),
 # ===================== ACT II : THE MASK =====================
 b("mask_becoming", "the_mask", "slow push-in, close on hands and face",
   "single bare bulb, cold, film grain", 8,
   "the man alone in a locked room building the creature, cracked bone-white greasepaint going onto his own face by hand for the first time, no happy grin, a wrong unsettling calm, single bare bulb, deep shadow, fully modest, cinematic realism",
   [("jack","I built him the only honest way, from the inside. The paint went on, and something answered."),("chorus","Not treatment. Not a costume. A becoming.")],
   "original dark theme, a cold piano and low strings, the mask entering, wholly original and transformative, instrumental only, no vocals, no lyrics",
   "a locked room tone, the small wet sound of paint, a slow breath"),
 b("mask_immersion", "the_mask", "static, the man in half-paint at a cluttered desk",
   "warm lamp and cold shadow, film grain", 8,
   "the man in half-applied paint living entirely as the character, a room papered with notes and images, no longer performing but inhabiting, a wrong unsettling calm, single lamp light, deep shadow, fully modest, cinematic realism",
   [("jack","I stopped performing him. I lived him, every waking hour, until his voice was easier to find than my own.")],
   "original dark theme, a circling cello motif over a faint machine pulse, immersion deepening, wholly original and transformative, instrumental only, no vocals, no lyrics",
   "a cluttered room tone, a pen scratching, a low steady hum"),
 b("mask_notebook", "the_mask", "insert, close on a handwritten page then eyes",
   "cold lamp on paper, film grain", 7,
   "close on a battered notebook filling with the character's thoughts in a cramped hand, then the man's hollowed eyes above it, no happy grin, obsessive quiet, single lamp light, deep shadow, fully modest, cinematic realism",
   [("jack","I kept his diary as if it were mine. Somewhere the two hands became one.")],
   "original dark theme, a spare piano and low drone, obsessive quiet, wholly original and transformative, instrumental only, no vocals, no lyrics",
   "a quiet room, a pen scratching, a page turning, a low hum"),
 b("mask_hotelroom", "the_mask", "wide, a small figure in a large dark room",
   "cold blue night, a single window glow, film grain", 8,
   "the man small in a large bare hotel room in the dead hours of night, awake and alone with the character, no happy grin, vast isolation, monumental negative space, single window light, fully modest, cinematic realism",
   [("jack","The nights grew long. A bare hotel room, the dark hours, and him, always awake beside me.")],
   "original dark theme, a lone piano over a cold sustained pad, sleepless isolation, wholly original and transformative, instrumental only, no vocals, no lyrics",
   "a large empty hotel room, a distant city at night, a clock, a slow breath"),
 b("mask_sleepless", "the_mask", "static, close on open eyes in the dark",
   "near-black with a thin cold light, film grain", 8,
   "close on the man's open sleepless eyes in near-darkness, hours crawling, the world thinned to a hum, no happy grin, hollow exhaustion, a thin cold light, deep shadow, fully modest, cinematic realism",
   [("jack","I could not sleep. And I learned what the sleepless already know, insomnia is only loneliness with the lights off."),("chorus","Insomnia is loneliness.")],
   "original dark theme, a faint ticking pulse under suspended strings, hollow and endless, wholly original and transformative, instrumental only, no vocals, no lyrics",
   "a dead-of-night room tone, a faint clock, a long uneven breath"),
 b("mask_thevoice", "the_mask", "slow push-in, half-lit face",
   "single hard key, deep black, film grain", 7,
   "the man in shadow discovering the character's laugh for the first time, a wrong sound that frightens even him, no happy grin, a chilling calm, single hard key light, deep black, fully modest, cinematic realism",
   [("jack","One night I found his laugh. It came out of my own mouth, and it frightened me.")],
   "original dark theme, a cold metallic resonance entering under the strings, wholly original and transformative, instrumental only, no vocals, no lyrics",
   "a quiet room, a low uneasy laugh stirring, a faint hum"),
 b("mask_body", "the_mask", "wide, a figure moving with a wrong new gait",
   "cold light, deep shadow, film grain", 7,
   "the man moving through a dim room with the character's wrong new gait and stillness, his own body no longer quite his, no happy grin, quiet unease, single hard key, generous negative space, fully modest, cinematic realism",
   [("jack","He got into my hands first. Then my walk. Then the way I held still. My body learned him before I noticed.")],
   "original dark theme, a low machine pulse under sparse strings, the body changing, wholly original and transformative, instrumental only, no vocals, no lyrics",
   "a dim room tone, slow deliberate footsteps, a low hum"),
 b("mask_losingedges", "the_mask", "static, a man half in paint half bare split by light",
   "half warm half cold, hard division, film grain", 8,
   "the man's face split by light, half bare and gentle, half painted and severe, the border between them dissolving, no happy grin, quiet horror, hard chiaroscuro, deep shadow, fully modest, cinematic realism",
   [("jack","I could no longer find the line where I ended and he began. Some mornings I did not look for it.")],
   "original dark theme, two motifs blurring into one dissonant line, wholly original and transformative, instrumental only, no vocals, no lyrics",
   "a quiet room, a low drone, a single slow breath"),
 b("mask_mirror", "the_mask", "static, a man before a mirror",
   "cold light, a dim mirror, film grain", 7,
   "the man before a mirror no longer recognising the face beneath the paint, the reflection a stranger wearing his features, no happy grin, quiet dread, single cold light, deep shadow, fully modest, cinematic realism",
   [("jack","I looked in the glass for the man I used to be. A stranger looked back, and smiled.")],
   "original dark theme, a cold piano and a thin high string, dread suspended, wholly original and transformative, instrumental only, no vocals, no lyrics",
   "a quiet room, a faint hum, a held breath"),
 b("mask_friends", "the_mask", "wide, a figure apart from blurred others",
   "cold blue, warm blur beyond, film grain", 7,
   "the man standing apart while blurred familiar people carry on without seeing him, only the mask visible to them now, no happy grin, isolation, generous negative space, fully modest, cinematic realism",
   [("jack","The people who loved me started talking to the mask. It answered better than I did.")],
   "original dark theme, receding strings and a lone cold piano, wholly original and transformative, instrumental only, no vocals, no lyrics",
   "distant muffled voices, a lonely room tone, a quiet breath"),
 # ===================== ACT II : THE UNRAVELING =====================
 b("unr_distance", "the_unraveling", "static two-shot, a widening gap",
   "cold blue, a single warm lamp failing, film grain", 8,
   "the luminous woman reaching gently toward the man and the man unable to let her close, a widening distance between them, no happy grin, ache and refusal, single failing warm light, deep shadow, fully modest, cinematic realism",
   [("julie","Talk to me. Wherever you have gone, come back. I am still here."),("jack","She reached for me, the way she always had. For the first time, I could not let her.")],
   "original dark theme, an aching solo violin over hushed strings, wholly original and transformative, instrumental only, no vocals, no lyrics",
   "a quiet room, soft rain, two breaths held apart", chars=["jack","julie"]),
 b("unr_reaching", "the_unraveling", "static two-shot, a hand not quite meeting",
   "cold blue, a single warm lamp, film grain", 8,
   "the luminous woman extending a hand toward the man across a small gap that will not close, his eyes turned away, no happy grin, tender futility, single warm light, deep shadow, fully modest, cinematic realism",
   [("julie","I do not need you to be alright. I only need you to let me in. Please."),("jack","She kept reaching. I kept telling her I was fine. It was the kindest lie I knew, and the cruelest.")],
   "original dark theme, an aching solo violin over hushed strings, a plea unanswered, wholly original and transformative, instrumental only, no vocals, no lyrics",
   "a quiet room, soft rain, a held breath, a hand lowering", chars=["jack","julie"]),
 b("unr_silence", "the_unraveling", "static, two figures back to back in shadow",
   "cold blue, deep shadow, film grain", 7,
   "the man and the woman in the same room yet turned from each other, a silence hardened to stone between them, no happy grin, grief and restraint, generous negative space, single cold light, fully modest, cinematic realism",
   [("jack","The silence I built to protect her became the wall that lost her.")],
   "original dark theme, a low sustained string and a single piano note, wholly original and transformative, instrumental only, no vocals, no lyrics",
   "a heavy silent room, a distant clock, two separate breaths", chars=["jack","julie"]),
 b("unr_triumph", "the_unraveling", "reverse from a bright premiere into black",
   "hot flashbulb white into deep black, film grain", 7,
   "the man in the painted role at a blazing premiere, the world crowning the performance, flashbulbs crashing, no happy grin, triumph gone hollow, single blazing key, deep black, fully modest, cinematic realism",
   [("jack","The world called it the performance of a lifetime. They did not know what it had cost the life."),("chorus","What the world adores, it rarely troubles to understand.")],
   "original dark theme, brass swelling into a hollow cold grandeur, wholly original and transformative, instrumental only, no vocals, no lyrics",
   "a roaring premiere, flashbulbs, a distant crowd, a single breath"),
 b("unr_aloneatpeak", "the_unraveling", "wide, a small figure in a vast dark space",
   "near-black, a thin cold light, film grain", 8,
   "the man alone in a vast dark room after the triumph, still in half-paint, adored by millions and utterly alone, no happy grin, monumental isolation, generous negative space, fully modest, cinematic realism",
   [("jack","That night I sat alone with the applause still ringing, and had never felt further from a single human being.")],
   "original dark theme, a lone cello over a cold pad, vast and empty, wholly original and transformative, instrumental only, no vocals, no lyrics",
   "a vast empty room, a faint ring of distant applause fading, a slow breath"),
 b("unr_exhaustion", "the_unraveling", "static, close on a worn face",
   "cold grey, deep shadow, film grain", 7,
   "close on the man's worn hollow face between takes, drained to the edge, the gift burning through the last of him, no happy grin, exhaustion, single soft key, deep shadow, fully modest, cinematic realism",
   [("jack","I was burning the candle from both ends and the middle. I mistook the light it gave for being alive.")],
   "original dark theme, a thin tired piano over a low drone, wholly original and transformative, instrumental only, no vocals, no lyrics",
   "a quiet room, a faint hum, a long weary breath"),
 b("unr_fadinglight", "the_unraveling", "slow push-in, warmth draining from a face",
   "warm fading to cold, film grain", 8,
   "close on the man as the last warmth drains from his face into a cold stillness, the gentle one receding beyond reach, no happy grin, quiet grief, single key light cooling, deep shadow, fully modest, cinematic realism",
   [("jack","I could feel the warmth going out of me, room by room, like a house closing for winter.")],
   "original dark theme, a warm line dimming into a cold sustained pad, wholly original and transformative, instrumental only, no vocals, no lyrics",
   "a quiet room, a low drone, a long slow breath"),
 b("unr_warnings", "the_unraveling", "static, blurred concerned faces, a sharp lonely man",
   "cold blue, film grain", 7,
   "blurred concerned faces at the edges of frame trying to reach the man who sits sharp and unreachable at the centre, no happy grin, quiet alarm unheeded, generous negative space, fully modest, cinematic realism",
   [("jack","People worried aloud. I smiled and told them the truth I believed, that I was fine, that I could stop him whenever I chose.")],
   "original dark theme, uneasy strings and a faint machine pulse, wholly original and transformative, instrumental only, no vocals, no lyrics",
   "muffled concerned voices, a lonely room tone, a quiet breath"),
 b("unr_lastordinary", "the_unraveling", "static, warm soft light on a gentle face",
   "warm amber, deep shadow, film grain", 8,
   "a rare moment of the gentle man surfacing through the exhaustion, bare-faced and quietly tender in warm light, the last clear glimpse of who he was, no paint, fragile warmth, single warm key, deep shadow, fully modest, cinematic realism",
   [("jack","There was one last ordinary day. I remember the light was kind. I did not know it was goodbye."),("chorus","The gentle one was still in there. He was only very tired.")],
   "original memoir theme returning warm and fragile, felt piano and cello, a tender ache, wholly original and transformative, instrumental only, no vocals, no lyrics",
   "a warm quiet room, soft afternoon sounds, a gentle breath"),
 b("unr_goodbye", "the_unraveling", "static two-shot, warm light between them",
   "warm amber failing to cold, film grain", 7,
   "the man and the luminous woman sharing an ordinary quiet moment, a goodbye neither of them knows to say, no paint, unbearable tenderness, single warm light, deep shadow, fully modest, cinematic realism",
   [("julie","Whatever it is, you can tell me. You know that. You always could."),("jack","I almost told her, that day. I almost said the word. I let it pass, the way I always had.")],
   "original memoir theme, intertwining cello and piano, a fragile unbearable warmth, wholly original and transformative, instrumental only, no vocals, no lyrics",
   "a quiet warm room, a clock, two soft breaths", chars=["jack","julie"]),
 b("unr_thewound", "the_unraveling", "slow push-in, half-lit face, paint returning",
   "cold light, paint half-on, film grain", 7,
   "the man alone as the paint goes back on, an old wound reopening behind his eyes, the gentle warmth of the day already receding, no happy grin, quiet grief hardening, single hard key, deep shadow, fully modest, cinematic realism",
   [("jack","But the day ended, as days do. And the thing I had let live in me was hungry again.")],
   "original dark theme, cold metallic resonance under a low swell, the mask returning, wholly original and transformative, instrumental only, no vocals, no lyrics",
   "a quiet room, a faint hum, the small wet sound of paint, a slow breath"),
 b("unr_prayer", "the_unraveling", "static, a bowed figure in a shaft of pale light",
   "near-black, a single pale shaft, film grain", 8,
   "the man bowed alone in a single pale shaft of light in a vast dark room, a wordless prayer for rest that he cannot name, no happy grin, exhausted reverence, monumental negative space, fully modest, cinematic realism",
   [("jack","I asked the dark for one honest night of sleep. I did not care, any longer, which side of morning I woke on."),("chorus","He only wanted to rest. The world heard the wrong wish.")],
   "original dark theme, a lone violin over a deep sub-bass, an aching plea, wholly original and transformative, instrumental only, no vocals, no lyrics",
   "a vast dark room, a faint low drone, a long uneven breath"),
 b("unr_thedark", "the_unraveling", "slow fade toward black on a still face",
   "collapsing to near-black, film grain", 7,
   "the man's face slowly swallowed by darkness, stillness settling, the world going out one light at a time, no happy grin, a quiet terrible calm, deep black closing in, fully modest, cinematic realism",
   [("jack","And then it all went dark. Not with a scream. With a quiet I mistook for peace.")],
   "original dark theme, everything receding to a single low sustained note, wholly original and transformative, instrumental only, no vocals, no lyrics",
   "a room going silent, a last faint hum, a slow final breath"),
 b("unr_midnight", "the_unraveling", "static wide, an empty chair in a shaft of light",
   "near-black, a single cold shaft, film grain", 8,
   "an empty chair in a shaft of cold light in a vast dark room, a low laugh stirring somewhere in the black where the man used to be, no figure clearly seen, dread and absence, monumental negative space, fully modest, cinematic realism",
   [("jack","At midnight, something laughed in the dark. And the man, the gentle one, the one she saw, never came back."),("chorus","Not treatment. Not a bandage. A metamorphosis.")],
   "original dark theme turning to dread, low strings and a faint heartbeat of percussion swelling as the last warm note collapses, wholly original and transformative, instrumental only, no vocals, no lyrics",
   "cold air, a low rising drone, a single distant echoing laugh"),
]

# ------------------------------------------------------------------ build shots
def build_shots():
    shots = []
    for be in BEATS:
        shots.append({
            "id": be["id"],
            "movement": be["mov"],
            "song": be["mov"],
            "chars": be["chars"],
            "dur": be["dur"],
            "mode": "A",
            "camera": be["cam"],
            "palette": be["pal"],
            "prompt": be["prompt"],
            "negative": NEG,
            "tone": TONE,
            "talking": False,
            "vo": True,
        })
    return shots

def main():
    confr = json.load(open(os.path.join(ROOT, "shots_confrontation.json")))
    confr_shots = confr["shots"]
    new_shots = build_shots()

    merged = {
        "film": "THE LAST LAUGH -- A Heath Ledger Tribute (full film)",
        "note": ("For Heath Ledger, 1979-2008. A tribute told through myth, not a reenactment. "
                 "Act I THE GIFT (the unseen man, being seen, the rise) and Act II THE MASK "
                 "(living the character until it consumed him -- immersion, insomnia, isolation) "
                 "lead into the existing confrontation as the climax. Jack is an original Ledger-order "
                 "figure, not a replica; the descent never depicts medication or a real death; the "
                 "Joker's fictional end stays the character's. Dignity before the film."),
        "globals": confr["globals"],
        "character_design": confr["character_design"],
        "characters": confr["characters"],
        "shots": new_shots + confr_shots,
    }
    json.dump(merged, open(os.path.join(ROOT, "shots_lastlaugh.json"), "w"), indent=2, ensure_ascii=False)

    # ---- script CSV ----
    with open(os.path.join(ROOT, "script", "lines_lastlaugh.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["shot_id", "character", "language", "text", "voice_description", "layer"])
        for be in BEATS:
            for sp, text in be["lines"]:
                layer = "chorus" if sp == "chorus" else "lead"
                name = {"jack": "Jack", "julie": "Julie", "chorus": "chorus"}[sp]
                w.writerow([be["id"], name, "en", text, vd(sp), layer])
    # append confrontation script rows (skip header)
    confr_lines = open(os.path.join(ROOT, "script", "lines_confrontation.csv")).read().splitlines()
    with open(os.path.join(ROOT, "script", "lines_lastlaugh.csv"), "a", newline="") as f:
        for ln in confr_lines[1:]:
            if ln.strip():
                f.write(ln + "\n")

    # ---- audio prompts CSV ----
    with open(os.path.join(ROOT, "audio", "prompts_lastlaugh.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["shot_id", "music_prompt", "ambience_prompt"])
        for be in BEATS:
            w.writerow([be["id"], be["music"], be["amb"]])
    confr_prompts = open(os.path.join(ROOT, "audio", "prompts_confrontation.csv")).read().splitlines()
    with open(os.path.join(ROOT, "audio", "prompts_lastlaugh.csv"), "a", newline="") as f:
        for ln in confr_prompts[1:]:
            if ln.strip():
                f.write(ln + "\n")

    # ---- config ----
    cfg = yaml.safe_load(open(os.path.join(ROOT, "config_confrontation.yaml")))
    cfg["film_title"] = "THE_LAST_LAUGH"
    cfg["shots_file"] = "shots_lastlaugh.json"
    cfg["paths"]["script"] = "script/lines_lastlaugh.csv"
    cfg["paths"]["audio_prompts"] = "audio/prompts_lastlaugh.csv"
    cfg["paths"]["audio_bg"] = "audio/bg_lastlaugh"
    cfg["paths"]["work"] = "/tmp/lastlaugh_work"
    cfg["paths"]["out"] = "output_lastlaugh"

    # score movements: new acts first (in order), then the confrontation's own
    new_movs = {}
    for mv in ["the_gift", "the_ascent", "the_mask", "the_unraveling"]:
        ids = [be["id"] for be in BEATS if be["mov"] == mv]
        new_movs[mv] = {"file": mv + ".mp3", "shots": ids}
    for mv in ["prelude", "search", "confrontation"]:
        new_movs[mv] = cfg["score"]["movements"][mv]
    cfg["score"]["movements"] = new_movs

    cfg["finale"]["recap_shots"] = ["gift_firstlight", "gift_theonewhosaw", "asc_applause",
                                     "mask_becoming", "unr_lastordinary", "pro_seen", "pro_silence",
                                     "confr_real", "confr_aim", "confr_sayit", "confr_smile"]
    cfg["finale"]["freeze_on"] = "confr_smile"
    cfg["active_profile"] = "lightning_a100_max"

    os.makedirs(os.path.join(ROOT, "audio", "bg_lastlaugh"), exist_ok=True)

    with open(os.path.join(ROOT, "config_lastlaugh.yaml"), "w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False, default_flow_style=False, allow_unicode=True, width=100000)

    # ---- report ----
    new_dur = sum(be["dur"] for be in BEATS)
    confr_dur = sum(s.get("dur", 0) for s in confr_shots)
    total = new_dur + confr_dur
    print("=== THE LAST LAUGH build ===")
    print("Act I+II new beats :", len(new_shots), "(", new_dur, "s )")
    print("Confrontation beats:", len(confr_shots), "(", confr_dur, "s )")
    print("TOTAL beats        :", len(merged["shots"]), " duration ~", total, "s = %.1f min" % (total / 60.0))
    print("movements          :", list(cfg["score"]["movements"].keys()))
    nlines = sum(len(be["lines"]) for be in BEATS)
    print("new script lines   :", nlines)
    print("active_profile     :", cfg["active_profile"])
    print("wrote: shots_lastlaugh.json, script/lines_lastlaugh.csv, audio/prompts_lastlaugh.csv, config_lastlaugh.yaml")


if __name__ == "__main__":
    main()
