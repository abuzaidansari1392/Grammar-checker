"""
Grammar Error Dataset Generator
Provides labeled training data for the ML error classifier.
"""

# Training samples: (sentence, error_type)
# Error types: subject_verb, tense, article, preposition, pronoun, spelling, punctuation, word_order, correct

TRAINING_DATA = [
    # Subject-Verb Agreement errors
    ("He go to school every day.", "subject_verb"),
    ("She don't like coffee.", "subject_verb"),
    ("They was playing football.", "subject_verb"),
    ("The dogs runs fast.", "subject_verb"),
    ("We was happy to see you.", "subject_verb"),
    ("He have a new car.", "subject_verb"),
    ("The team are playing well.", "subject_verb"),
    ("She were at the party.", "subject_verb"),
    ("John and Mary was friends.", "subject_verb"),
    ("The children plays outside.", "subject_verb"),
    ("My friend don't know the answer.", "subject_verb"),
    ("Everyone have their own opinion.", "subject_verb"),
    ("Neither of them are correct.", "subject_verb"),
    ("The news are shocking.", "subject_verb"),
    ("Mathematics are my favorite subject.", "subject_verb"),

    # Tense errors
    ("Yesterday, I go to the market.", "tense"),
    ("She will went to the store.", "tense"),
    ("He has went to Paris.", "tense"),
    ("I have saw that movie.", "tense"),
    ("They are play cricket now.", "tense"),
    ("We had went there before.", "tense"),
    ("She will has finished by noon.", "tense"),
    ("He was go to work.", "tense"),
    ("I have ate breakfast.", "tense"),
    ("They has been waiting.", "tense"),
    ("She will going home soon.", "tense"),
    ("He has ran five miles.", "tense"),
    ("We are went to the park.", "tense"),
    ("I will have went there.", "tense"),
    ("She was ran quickly.", "tense"),

    # Article errors
    ("She is a honest woman.", "article"),
    ("He bought the umbrella yesterday.", "article"),
    ("I need a information about this.", "article"),
    ("She ate an apple and the orange.", "article"),
    ("He is a university professor.", "article"),  # correct usage
    ("I have a hour to spare.", "article"),
    ("She is the best in a class.", "article"),
    ("He went to an hospital.", "article"),
    ("I need the advice from you.", "article"),
    ("She is a European student.", "article"),
    ("He read an useful book.", "article"),
    ("The sun rises in an east.", "article"),
    ("She has the flu.", "article"),
    ("He is going to an university.", "article"),
    ("I saw a elephant at the zoo.", "article"),

    # Preposition errors
    ("She is interested on music.", "preposition"),
    ("He is good in cooking.", "preposition"),
    ("I will meet you on Monday at noon.", "preposition"),  # correct
    ("She is afraid from dogs.", "preposition"),
    ("He is married with a doctor.", "preposition"),
    ("I am angry on him.", "preposition"),
    ("She arrived to the station.", "preposition"),
    ("He is jealous from his brother.", "preposition"),
    ("I am bored of waiting.", "preposition"),
    ("She is tired from work.", "preposition"),
    ("He is different of his brother.", "preposition"),
    ("I am looking forward for meeting you.", "preposition"),
    ("She is responsible of the project.", "preposition"),
    ("He is dependent of his parents.", "preposition"),
    ("I am worried for my health.", "preposition"),

    # Pronoun errors
    ("Him went to the store.", "pronoun"),
    ("Me and my friend went there.", "pronoun"),
    ("The dog hurt his self.", "pronoun"),
    ("Her is a good student.", "pronoun"),
    ("Between you and I, this is wrong.", "pronoun"),
    ("Us students need more time.", "pronoun"),
    ("Them children are playing.", "pronoun"),
    ("He gave it to she.", "pronoun"),
    ("Me am happy.", "pronoun"),
    ("The teacher praised John and I.", "pronoun"),
    ("Him and her went shopping.", "pronoun"),
    ("Who did you speak to? (formal)", "pronoun"),
    ("Us need to talk.", "pronoun"),
    ("He hurt hisself.", "pronoun"),
    ("They helped ourselves.", "pronoun"),

    # Spelling errors
    ("She recieved a letter today.", "spelling"),
    ("He beleives in hard work.", "spelling"),
    ("I occured to me that she left.", "spelling"),
    ("The buisness is doing well.", "spelling"),
    ("She is a beautifull woman.", "spelling"),
    ("He achived his goal.", "spelling"),
    ("I definately agree with you.", "spelling"),
    ("She is responsable for this.", "spelling"),
    ("He seperated from his partner.", "spelling"),
    ("I am greatful for your help.", "spelling"),
    ("The goverment passed a new law.", "spelling"),
    ("She is embarassed by the mistake.", "spelling"),
    ("He recomended a good book.", "spelling"),
    ("I neccessarily disagree.", "spelling"),
    ("She had a wierd feeling.", "spelling"),

    # Punctuation errors
    ("Lets go to the park today", "punctuation"),
    ("She said i am happy.", "punctuation"),
    ("He went home, and she stayed.", "punctuation"),  # correct
    ("Where are you going.", "punctuation"),
    ("Its a beautiful day outside.", "punctuation"),
    ("She bought apples, oranges and bananas", "punctuation"),
    ("He said that he would come but he didnt.", "punctuation"),
    ("Dont forget your umbrella.", "punctuation"),
    ("She isnt coming to the party.", "punctuation"),
    ("The meeting is on monday morning.", "punctuation"),
    ("He wont be able to attend.", "punctuation"),
    ("I cant believe you said that", "punctuation"),
    ("Shes going to the store tomorrow.", "punctuation"),
    ("Whos coming to the party?", "punctuation"),
    ("Theyre not ready yet.", "punctuation"),

    # Word Order errors
    ("Always she comes late.", "word_order"),
    ("He yesterday went to school.", "word_order"),
    ("I have never been to Paris.", "word_order"),  # correct
    ("Often they play football on weekends.", "word_order"),
    ("She quickly ran.", "word_order"),
    ("I much very like this movie.", "word_order"),
    ("He to school goes every day.", "word_order"),
    ("Yesterday came she home late.", "word_order"),
    ("A red beautiful car she drives.", "word_order"),
    ("He has never to Paris been.", "word_order"),
    ("Only she knows the answer.", "word_order"),  # can be correct depending on context
    ("He gave her friend a gift beautiful.", "word_order"),
    ("She reads the always newspaper.", "word_order"),
    ("I do like very much this.", "word_order"),
    ("He home goes at five.", "word_order"),

    # More Subject-Verb samples
    ("The boy run to the store.", "subject_verb"),
    ("Each of the students have a book.", "subject_verb"),
    ("My sisters was at the mall.", "subject_verb"),
    ("The teacher give homework daily.", "subject_verb"),
    ("He don't understand the question.", "subject_verb"),
    ("She have many friends.", "subject_verb"),
    ("They was very happy.", "subject_verb"),
    ("The cat chase the mouse.", "subject_verb"),
    ("We was eating lunch.", "subject_verb"),
    ("She go to the gym.", "subject_verb"),

    # More Tense samples
    ("She will went there last night.", "tense"),
    ("He have finish his work.", "tense"),
    ("I seen that film last week.", "tense"),
    ("She is went to the store.", "tense"),
    ("He will completed the task.", "tense"),
    ("They have go to the park.", "tense"),
    ("I am saw the movie yesterday.", "tense"),
    ("She has go to the meeting.", "tense"),
    ("He was ran very fast.", "tense"),
    ("We have did our homework.", "tense"),

    # More Article samples
    ("He is a excellent student.", "article"),
    ("She bought an book at the store.", "article"),
    ("I need an pen to write.", "article"),
    ("He is an honest person.", "article"),
    ("She has a umbrella.", "article"),
    ("He drove an car to work.", "article"),
    ("I saw an dog in the park.", "article"),
    ("She is a intelligent woman.", "article"),
    ("He bought a orange at the market.", "article"),
    ("It was an historical event.", "article"),

    # More Preposition samples
    ("She is good on playing piano.", "preposition"),
    ("He is interested of reading.", "preposition"),
    ("I am angry with her.", "preposition"),  # correct
    ("She is bored from watching TV.", "preposition"),
    ("He is fond in music.", "preposition"),
    ("She is afraid of the dark.", "preposition"),  # correct
    ("I am tired from studying.", "preposition"),
    ("He is jealous of his friend.", "preposition"),  # correct
    ("She is married from a teacher.", "preposition"),
    ("He depends of his family.", "preposition"),

    # More Pronoun samples
    ("Me and John went to school.", "pronoun"),
    ("The teacher called he and I.", "pronoun"),
    ("Her and him are good friends.", "pronoun"),
    ("Us children love to play.", "pronoun"),
    ("The gift was for she.", "pronoun"),
    ("Between you and I the answer is clear.", "pronoun"),
    ("Them are the best students.", "pronoun"),
    ("Who did you talk to?", "pronoun"),  # borderline — keep
    ("He bought hisself a new car.", "pronoun"),
    ("She hurt herself while cooking.", "pronoun"),  # correct

    # More Spelling samples
    ("The whether is very hot today.", "spelling"),
    ("She is studing for her exam.", "spelling"),
    ("He has alot of homework.", "spelling"),
    ("I recieve many emails daily.", "spelling"),
    ("She is a wonderfull teacher.", "spelling"),
    ("He writting a letter.", "spelling"),
    ("I beleive in your ability.", "spelling"),
    ("She is responsibe for the task.", "spelling"),
    ("He occured to me later.", "spelling"),
    ("The buisness was very profitable.", "spelling"),

    # More Punctuation samples
    ("She said its going to rain.", "punctuation"),
    ("He couldnt come to the party.", "punctuation"),
    ("Dont worry about it.", "punctuation"),
    ("Shes a very good student.", "punctuation"),
    ("The cats bowl is empty.", "punctuation"),
    ("Whos at the door", "punctuation"),
    ("He wont be coming today.", "punctuation"),
    ("They wouldnt listen to advice.", "punctuation"),
    ("Its a nice day outside.", "punctuation"),
    ("She hasnt finished yet.", "punctuation"),

    # More Word Order samples
    ("She quickly very ran.", "word_order"),
    ("He to work goes every day.", "word_order"),
    ("I much like this very.", "word_order"),
    ("She yesterday went shopping.", "word_order"),
    ("He a big beautiful house has.", "word_order"),
    ("Always he is late for class.", "word_order"),
    ("She reads the newspaper always.", "word_order"),
    ("He gave a gift beautiful.", "word_order"),
    ("I never have been there.", "word_order"),
    ("She only knows the truth.", "word_order"),  # context-dependent

    # Correct sentences (no error)
    ("She goes to school every day.", "correct"),
    ("He has been working here for five years.", "correct"),
    ("I would like a cup of coffee, please.", "correct"),
    ("They are interested in learning new things.", "correct"),
    ("The children played in the park yesterday.", "correct"),
    ("She received an important email this morning.", "correct"),
    ("He is responsible for the project.", "correct"),
    ("I am grateful for your assistance.", "correct"),
    ("The government passed new legislation.", "correct"),
    ("She doesn't like spicy food.", "correct"),
    ("We were watching television last night.", "correct"),
    ("He will have finished the report by tomorrow.", "correct"),
    ("I am looking forward to meeting you.", "correct"),
    ("The team has won three consecutive matches.", "correct"),
    ("She was embarrassed by the mistake.", "correct"),
    ("He recommended a very interesting book.", "correct"),
    ("I definitely agree with your opinion.", "correct"),
    ("The business is performing well this quarter.", "correct"),
    ("She is married to a doctor.", "correct"),
    ("He believes in the power of education.", "correct"),
]


def get_training_data():
    """Return training data as parallel lists."""
    sentences = [item[0] for item in TRAINING_DATA]
    labels = [item[1] for item in TRAINING_DATA]
    return sentences, labels


def get_label_mapping():
    """Return label-to-index and index-to-label mappings."""
    labels = list(set(item[1] for item in TRAINING_DATA))
    labels.sort()
    label_to_idx = {label: idx for idx, label in enumerate(labels)}
    idx_to_label = {idx: label for label, idx in label_to_idx.items()}
    return label_to_idx, idx_to_label


ERROR_DESCRIPTIONS = {
    "subject_verb": "Subject-Verb Agreement Error",
    "tense": "Tense Error",
    "article": "Article Error",
    "preposition": "Preposition Error",
    "pronoun": "Pronoun Error",
    "spelling": "Spelling Error",
    "punctuation": "Punctuation Error",
    "word_order": "Word Order Error",
    "correct": "No Error Detected",
}

ERROR_EXPLANATIONS = {
    "subject_verb": "The subject and verb must agree in number. Singular subjects take singular verbs; plural subjects take plural verbs.",
    "tense": "Verb tense must be consistent and appropriate for the time being described.",
    "article": "Use 'a' before consonant sounds and 'an' before vowel sounds. 'The' is used for specific nouns.",
    "preposition": "Prepositions show relationships between words. Common ones include: in, on, at, by, for, with, about.",
    "pronoun": "Pronouns must agree with their antecedent in number, gender, and case.",
    "spelling": "Check the spelling of words — common mistakes include silent letters and vowel patterns.",
    "punctuation": "Correct punctuation is essential for clarity. Check apostrophes, commas, and sentence-ending marks.",
    "word_order": "English follows Subject-Verb-Object order. Adjectives usually come before nouns.",
    "correct": "The sentence appears grammatically correct.",
}
