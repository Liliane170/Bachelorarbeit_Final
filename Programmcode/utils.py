def decode_outputs(predicted_labels):
    entities = []

    for _, elem in enumerate(predicted_labels):
        attach = False

        if not elem["word"].startswith("Ġ"):
            attach = True

        if attach:
            entities[-1]["word"] += elem["word"]
            entities[-1]["end"] = elem["end"]
        else:
            entities.append(
                {
                    "word": elem["word"],
                    "start": elem["start"],
                    "end": elem["end"],
                    "entity": elem["entity"],
                }
            )

    
    for elem in entities:
        elem["word"] = elem["word"][1:]

    return entities