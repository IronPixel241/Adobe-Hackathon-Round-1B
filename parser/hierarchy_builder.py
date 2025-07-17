def build_hierarchy(blocks):
    # Improved title detection: pick top font-size block from page 1
    first_page_blocks = [b for b in blocks if b['page'] == 1]
    if first_page_blocks:
        top_title_block = max(first_page_blocks, key=lambda x: x['font_size'])
        title = top_title_block['text'].strip()
    else:
        title = ""
    seen = set()
    outline = []
    for block in blocks:
        heading_level = block.get("level")
        if heading_level in ["H1", "H2", "H3", "H4"]:
            key=block['text'].strip();
            if(key in seen):
              continue  
            seen.add(key)
            outline.append({
                "level": heading_level,
                "text": block["text"].strip(),  # keep trailing punctuation
                "page": block["page"]
            })

    return {"title": title, "outline": outline}