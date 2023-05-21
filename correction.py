from spellchecker import SpellChecker


def correct_text_final(text, text_language="ru"):
    speller = SpellChecker(language=text_language)
    old_text_version = text.split()
    grammar_fixed_text = []
    punctual_symbols = [',', '.', '?', '-', '!', '...', ';', ':']
    mistake_counter = 0
    for word in text.split():
        if word in punctual_symbols:
            grammar_fixed_text.append((word, 0))
        if speller.correction(word) is None:
            grammar_fixed_text.append((word, 0))
        else:
            grammar_fixed_text.append((speller.correction(word), 1))
            mistake_counter += 1
    return (grammar_fixed_text, mistake_counter)
