import seaborn as sns


def rgb2hex(r, g, b):
    return "#{:02x}{:02x}{:02x}".format(int(r * 255), int(g * 255), int(b * 255))


def make_color_palette(labels):
    color_palette = sns.color_palette(n_colors=len(labels))
    color_map = {x: rgb2hex(*y) for x, y in zip(labels, color_palette)}
    return color_map


def hf_ents_to_displacy_format(ents, ignore_entities=[]):

    s_ents = {}
    s_ents["text"] = " ".join([e["word"] for e in ents])
    spacy_ents = []
    start_pointer = 0
    if "entity_group" in ents[0]:
        entity_key = "entity_group"
    else:
        entity_key = "entity"
    for i, ent in enumerate(ents):
        if ent[entity_key] not in ignore_entities:
            spacy_ents.append(
                {
                    "start": start_pointer,
                    "end": start_pointer + len(ent["word"]),
                    "label": ent[entity_key],
                }
            )
        start_pointer = start_pointer + len(ent["word"]) + 1
    s_ents["ents"] = spacy_ents
    s_ents["title"] = None
    return s_ents
