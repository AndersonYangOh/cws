# -*- coding: utf-8 -*-

import re

SENT_DELIM = u"""。|？|！"""

def clean_and_separate(s):
    s = re.sub(r"\[|]/nz", "", s)
    s = re.sub(r"\/[a-z0-9]+", "", s)
    s = unicode(s.decode("utf8"))

    # match sentence deliminters
    matches = re.finditer(SENT_DELIM, s)
    ends = [m.end() for m in matches]

    sents = []
    start = 0
    for end in ends:
        sent = s[start:end]
        sent = [c for c in sent if c != " "]
        sent = " ".join(sent).encode("utf8")
        sents.append(sent)
        start = end
    return sents

def clean_tags(s):
    s = re.sub(r"\[|]/nz", "", s)
    s = re.sub(r"\/[a-z0-9]+", "", s)
    s = unicode(s.decode("utf8"))

    # match sentence deliminters
    matches = re.finditer(SENT_DELIM, s)
    ends = [m.end() for m in matches]

    sents = []
    start = 0
    for end in ends:
        sent = s[start:end]
        phrases = sent.split(" ")
        if len(phrases) == 0:
            continue
        words, tags = [], []
        for phrase in phrases:
            if len(phrase) == 0:
                continue
            if len(phrase) == 1:
                words.append(phrase)
                tags.append("S")
            else:
                words.append(phrase[0])
                tags.append("B")
                for word in phrase[1:len(phrase) - 1]:
                    words.append(word)
                    tags.append("I")
                words.append(phrase[-1])
                tags.append("E")
        sents.append((words, tags))

    return sents