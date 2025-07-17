import re
from copy import deepcopy
from utils import decode_outputs
import math
from word2number import w2n
import random
import json


class Anonymizer:
    def __init__(self, classifier):

        random.seed(42)

        self.classifier = classifier


        with open("months.txt", "r") as f: 
            self.months = f.readlines()
            self.months = [m.replace("\n", "") for m in self.months]

        with open("written_numbers.txt", "r") as f: 
            self.written_numbers = f.readlines()
            self.written_numbers = [w.replace("\n", "") for w in self.written_numbers]

        self.valid_surrounding_chars = [
            ".",
            ",",
            ";",
            "!",
            ":",
            "\n",
            "’",
            "‘",
            "'",
            '"',
            "?",
            "-",
        ]
        
        self.pronoun_map_reverse = {
            "he": "PRONOUN_1",
            "she": "PRONOUN_2",
            "him": "PRONOUN_3",
            "his": "PRONOUN_4",
            "her": "PRONOUN_5",
            "hers": "PRONOUN_6",
            "himself": "PRONOUN_7",
            "herself": "PRONOUN_8",
            "mr": "MR/MS_1",
            "mrs": "MR/MS_2",
            "mr.": "MR/MS_3",
            "mrs.": "MR/MS_4",
            "miss": "MR/MS_5",
            "ms": "MR/MS_6",
            "dr": "TITLE_1",
            "dr.": "TITLE_2",
            "prof": "TITLE_3",
            "prof.": "TITLE_4",
            "sir": "TITLE_5",
            "dame": "TITLE_6",
            "madam": "TITLE_7",
            "lady": "TITLE_8",
            "lord": "TITLE_9",
        }

        self.numeric_date = {}

        self.url = {}
        
        self.category_map = {}


    # Hilfsfunktionen für Tagging und NER

    def replace_identified_entities(self, entities, anon_input_seq, entity2generic):
        
        for phrase, _ in sorted(
            entities.items(), key=lambda x: len(x[0]), reverse=True
        ):
            if len(phrase) > 1 or phrase.isalnum():
                try:
                    for char in self.valid_surrounding_chars:
                        anon_input_seq = re.sub(
                            "[^a-zA-Z0-9]{}[{}]".format(phrase, char),
                            " {}{}".format(entity2generic[phrase], char),
                            anon_input_seq,
                        )

                    anon_input_seq = re.sub(
                        "[^a-zA-Z0-9\n]{}[^a-zA-Z0-9\n]".format(phrase),
                        " {} ".format(entity2generic[phrase]),
                        anon_input_seq,
                    )

                    anon_input_seq = re.sub(
                        "[\n]{}".format(phrase),
                        "\n{}".format(entity2generic[phrase]),
                        anon_input_seq,
                    )

                    anon_input_seq = re.sub(
                        "{}".format(phrase),
                        "{}".format(entity2generic[phrase]),
                        anon_input_seq,
                    )
                except re.error:
                    anon_input_seq = anon_input_seq.replace(
                        "{}".format(phrase), "{}".format(entity2generic[phrase])
                    )

        return anon_input_seq

    def get_identifiable_tokens(self, text_input):
        predictions = decode_outputs(
            self.classifier(text_input)
        )

        entities = {
            p["word"]: p["entity"]
            for p in predictions
            if p["entity"] != "NONE" and len(p["word"]) > 1 and p["word"].isalnum()
        }

        return entities

    def get_entity_type_mapping(self, entities):
        entity2generic_c = {v: 1 for _, v in entities.items()}
        entity2generic = {}

        for phrase, entity_type in entities.items():
            entity2generic[phrase] = "{}_{}".format(
                entity_type, entity2generic_c[entity_type]
            )

            entity2generic_c[entity_type] += 1

        return entity2generic

    def replace_numerics(self, anon_input_seq):
        # https://pythonexamples.org/python-regex-extract-find-all-the-numbers-in-string/
        all_numeric = list(set(re.findall("[0-9]+", anon_input_seq)))
        numeric_map = {k: "NUMERIC_{}".format(v + 1) for v, k in enumerate(all_numeric)}

        for k, v in sorted(numeric_map.items(), key=lambda x: int(x[0]), reverse=True):
            anon_input_seq = re.sub(
                "[^NUMERIC_0-9+]{}".format(k), " {}".format(v), anon_input_seq
            )

        return anon_input_seq, numeric_map

    def replace_pronouns(self, anon_input_seq, ner):

        # https://blog.hubspot.com/marketing/gender-neutral-pronouns
        pronoun_map_anonymize = {
            "he": "PRONOUN",
            "she": "PRONOUN",
            "him": "PRONOUN",
            "his": "PRONOUN",
            "her": "PRONOUN",
            "hers": "PRONOUN",
            "himself": "PRONOUN",
            "herself": "PRONOUN",
            "mr": "MR/MS",
            "mrs": "MR/MS",
            "mr.": "MR/MS",
            "mrs.": "MR/MS",
            "miss": "MR/MS",
            "ms": "MR/MS",
            "dr": "TITLE",
            "dr.": "TITLE",
            "prof": "TITLE",
            "prof.": "TITLE",
            "sir": "TITLE",
            "dame": "TITLE",
            "madam": "TITLE",
            "lady": "TITLE",
            "lord": "TITLE",

        }
        
        pronoun_map = self.pronoun_map_reverse if ner else pronoun_map_anonymize  

        for k, v in pronoun_map.items():
            if anon_input_seq.startswith("{} ".format(k)):
                anon_input_seq = anon_input_seq.replace(
                    "{} ".format(k), "{} ".format(v), 1
                )

            if anon_input_seq.startswith("{} ".format(k.capitalize())):
                anon_input_seq = anon_input_seq.replace(
                    "{} ".format(k.capitalize()), "{} ".format(v), 1
                )

            for char in self.valid_surrounding_chars:
                anon_input_seq = re.sub(
                    "[^a-zA-Z0-9]{}[{}]".format(k, char),
                    " {}{}".format(v, char),
                    anon_input_seq,
                )
                anon_input_seq = re.sub(
                    "[^a-zA-Z0-9]{}[{}]".format(k.capitalize(), char),
                    " {}{}".format(v, char),
                    anon_input_seq,
                )

            anon_input_seq = re.sub(
                "[^a-zA-Z0-9]{}[^a-zA-Z0-9]".format(k),
                " {} ".format(v),
                anon_input_seq,
            )
            anon_input_seq = re.sub(
                "[^a-zA-Z0-9]{}[^a-zA-Z0-9]".format(k.capitalize()),
                " {} ".format(v),
                anon_input_seq,
            )

        return anon_input_seq

    def replace_numbers_and_months(self, anon_input_seq, ner = False):
        entity2generic_c = {"MONTH": 1, "NUMBER": 1} if ner else {"DATE": 1, "NUMERIC": 1}
        entity2generic = {}

        spl = re.split("[ ,.-]", anon_input_seq)

        for word in spl:
            if word.lower() in self.written_numbers:
                try:
                    _ = entity2generic[word]
                except KeyError:
                    entity2generic[word] = "{}_{}".format(
                        ("NUMBER" if ner else "NUMERIC"), (entity2generic_c["NUMBER"] if ner else entity2generic_c["NUMERIC"])
                    )
                    if ner:
                        entity2generic_c["NUMBER"] += 1
                    else:
                        entity2generic_c["NUMERIC"] += 1

        for word in spl:
            if word.lower() in self.months:
                try:
                    _ = entity2generic[word]
                except KeyError:
                    entity2generic[word] = "{}_{}".format(
                        ("MONTH" if ner else "DATE"), (entity2generic_c["MONTH"] if ner else entity2generic_c["DATE"])
                    )
                    if ner:
                        entity2generic_c["MONTH"] += 1
                    else:
                        entity2generic_c["DATE"] += 1
      

        self.numeric_date = entity2generic

        for phrase, replacement in sorted(
            entity2generic.items(), key=lambda x: len(x[0]), reverse=True
        ):
            for char in self.valid_surrounding_chars:
                anon_input_seq = re.sub(
                    "[^a-zA-Z0-9]{}[{}]".format(phrase, char),
                    " {}{}".format(replacement, char),
                    anon_input_seq,
                )

            anon_input_seq = re.sub(
                "[^a-zA-Z0-9]{}[^a-zA-Z0-9]".format(phrase),
                " {} ".format(replacement),
                anon_input_seq,
            )

        return anon_input_seq
    
    def inject_original_phrases(self, anonymized_text, entity2generic):
        """
        Ersetzt alle Platzhalter im mit Tagging anonymisierten Text durch {'Original': 'Platzhalter'},
        aber nur bei exaktem Worttreffer (keine Teilstrings).
        """
        generic2entity = {v: k for k, v in entity2generic.items()}

        for generic, original in sorted(generic2entity.items(), key=lambda x: len(x[0]), reverse=True):
            pattern =  re.escape(generic) 

            def replacement_function(value):
                return f"{{'{original}': '{generic}'}}"

            anonymized_text = re.sub(pattern, replacement_function, anonymized_text)

        return anonymized_text

    
    def replace_with_mapping(self, text, pattern, base_placeholder):

        b = re.findall(pattern, text)

        general_map = {}
        i = 1
        for a in b:
            if a not in general_map:
                general_map[a] = f"{base_placeholder}_{i}"
                i += 1
            text = text.replace(a, general_map[a])

        return text, general_map

    # Hilfsfunktionen für GENERALIZATION und RANDOMISATION

    def replacer(self, bin_func, true_female_pronouns = None, true_male_pronouns = None, entity = None):
        """
        Gibt eine Funktion zurück, die in re.sub als Ersetzungsfunktion verwendet werden kann.
        """
        def _inner(match):
            value_str = match.group(1)
            type_label = match.group(2)

            binned = False if bin_func == None else (bin_func(value_str) if (true_female_pronouns == None and entity == None) else (bin_func(value_str, true_female_pronouns, true_male_pronouns) if entity == None else bin_func(value_str, entity)))
    
            if binned:
                return binned
            else:
                return f"[{type_label}]"
        return _inner

    def getRegex(self, entity_type):
        
        pattern = rf"\{{\s*'([^']+)'\s*:\s*'({entity_type})(?:_\d+)?'\s*\}}"

        return pattern


    #Hilfsfunktionen für GENERALIZATION 
    
    def generate_interval(self, value_str):
        """Konvertiert eine Zahl in ein Intervall wie [1-10], [11-20], ..."""
        try:
            value = float(value_str)
            if value < 0:
                return "[<0]"
            elif value < 1:
                return "[0-1]"
            elif value < 1000:
                lower = (math.floor(value / 10) * 10) + 1
                upper = lower + 9
                return f"[{lower}-{upper}]"
            else:
                lower = (math.floor(value / 100) * 100) + 1
                upper = lower + 999
                return f"[{lower}-{upper}]"
        except ValueError:
            try: 
                value = w2n.word_to_num(value_str)
                if value < 0:
                    return "[<0]"
                elif value < 1:
                    return "[0-1]"
                elif value < 1000:
                    lower = (math.floor(value / 10) * 10) + 1
                    upper = lower + 9
                    return f"[{lower}-{upper}]"
                else:
                    lower = (math.floor(value / 100) * 100) + 1
                    upper = lower + 999
                    return f"[{lower}-{upper}]"
            except ValueError:
                return None
    
    def generalize_url(self, value_str):
        
        base_url = re.match(r'^https?://[^/]+', value_str).group()
        
        return f"[{base_url}]"
    
    def gen_mail(self, value_str):
        
        match = re.search(r'@[\w.-]+\.[A-Za-z]{2,}', value_str)
        
        if match:
            return f"[{match.group()}]"
        return None
        
    
    def generalize_pronoun(self, value_str):
        
        
        if value_str in ("she", "he"):
            return "[she/he]"
        elif value_str in ("her", "him", "his", "hers"):
            return "[her/him/his/hers]"
        elif value_str in ("herself", "himself"):
            return "[herself/himself]"
        return None

    # Hilfsfunktionen für RANDOMISATION
    
    def pronoun_randomize(self, value_str, true_female_pronouns, true_male_pronouns):
        
        
        if (value_str == "she" and true_female_pronouns == 1) or (value_str == "he" and true_male_pronouns == 0):
            return "[she]"
        elif (value_str == "he" and true_male_pronouns == 1) or (value_str == "she" and true_female_pronouns == 0):
            return "[he]"
        elif (value_str in ("her", "hers") and true_female_pronouns == 1) or (value_str in ("him", "his") and true_male_pronouns == 0):
            return "[her/hers]"
        elif (value_str in ("him", "his") and true_male_pronouns == 1) or (value_str in ("her", "hers") and true_female_pronouns == 0):
            return "[him/his]"
        elif (value_str == "herself" and true_female_pronouns == 1) or (value_str == "himself" and true_male_pronouns == 0):
            return "[herself]"
        elif (value_str == "himself" and true_male_pronouns == 1) or (value_str == "herself" and true_female_pronouns == 0):
            return "[himself]"
        return None
        
    def mr_mrs_randomize(self, value_str, true_female_pronouns, true_male_pronouns):
  
            if (value_str in ("mrs", "mrs.", "miss", "ms") and true_female_pronouns == 1) or (value_str in ("mr", "mr.") and true_male_pronouns == 0):
                return f"[{random.choice(["mrs", "miss", "ms"])}]"
            elif (value_str in ("mr", "mr.") and true_male_pronouns == 1) or (value_str in ("mrs", "mrs.", "miss", "ms") and true_female_pronouns == 0):
                return f"[{random.choice(["mr"])}]"
            return None
    
    
    def title_randomize(self, value_str, true_female_pronouns, true_male_pronouns):
  
            if (value_str in ("dame", "madam", "lady") and true_female_pronouns == 1) or (value_str in ("lord", "sir") and true_male_pronouns == 0):
                return f"[{random.choice(["dame", "madam", "lady", "dr", "prof"])}]"
            elif (value_str in ("lord", "sir") and true_male_pronouns == 1) or (value_str in ("dame", "madam", "lady") and true_female_pronouns == 0):
                return f"[{random.choice(["lord", "sir", "dr", "prof"])}]"
            elif value_str in ("prof", "prof.", "dr", "dr."):
                return f"[{random.choice(["dame", "madam", "lady", "dr", "prof", "lord", "sir"])}]"
            return None
    
    def numeric_randomize(self, value_str, entity):

        try:

            if entity != 'AGE' and entity != 'NUMBER':
                value = float(value_str)

            if entity == 'AGE':
                number = random.randint(1, 100)
            elif entity == 'ADDRESS':
                number = random.randint(1, 100)
            elif entity == 'NUMBER':
                number = random.randint(1, 20)
            elif entity == 'NUMERIC':
                if value < 0:
                    number = random.randint(-1, -100)
                elif value < 100:
                    number = random.randint(1, 100)
                else:
                    number = random.randint(1, 10000)
            elif entity == 'DATE':
                if value < 13:
                    number = random.randint(1, 12)
                elif value < 32:
                    number = random.randint(1, 28)
                else:
                    number = random.randint(1950, 2025)
            else:
                return None

            return f"[{str(number)}]"
        
        except ValueError:
                return None

     
    def category_replacer(self, category, username = False):

        def _inner(match):
            value = match.group(0)
            if value not in self.category_map:
                with open("random.json", "r") as f:
                    data = json.load(f)
                new_value = f"[{random.choice(data[category])}]" if username == False else self.generate_random_username()
                self.category_map[value] = new_value
            return self.category_map[match.group(0)]
        return _inner
        
    
    def generate_random_url(self, value_str):
        with open("random.json", "r") as f:
                    data = json.load(f)
        domain = random.choice(data["domains"])
        path = ''.join(random.choices('abcdefghijklmnopqrstuvwxyz0123456789', k=8))
        return f"[https://{domain}/{path}]"
    
    def generate_random_email(self, value_str):
        with open("random.json", "r") as f:
                    data = json.load(f)
        firstname = random.choice(data["first_names"])
        lastname = random.choice(data["last_names"])
        domain = random.choice(data["email_domains"])
        
        return f"[{firstname}.{lastname}@{domain}]"
    
    def generate_random_username(self):
        with open("random.json", "r") as f:
                    data = json.load(f)
        name = random.choice(data["first_names"])
        number = random.randint(1, 1000)
        return f"[u/{name}{number}]"

    # Hauptfunktionen für NER, TAGGING, GENERALIZATION, RANDOMISATION und SUPPRESSION
    

    def tag(self, input_seq, selected_entities=None):
        orig_input_seq = deepcopy(input_seq)

        entities = self.get_identifiable_tokens(deepcopy(input_seq))
        

        entity2generic = self.get_entity_type_mapping(entities)
       

        anon_input_seq = re.sub(r"https?://\S+", "URL", orig_input_seq)
        
        anon_input_seq = re.sub(r"u\/[A-Za-z0-9_-]+", "PERSON_USERNAME", anon_input_seq)
        
        anon_input_seq = re.sub(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b", "EMAIL_ADDRESS", anon_input_seq)

        anon_input_seq = self.replace_identified_entities(
            entities, anon_input_seq, entity2generic
        )

        anon_input_seq, numeric_map = self.replace_numerics(anon_input_seq)

        anon_input_seq = self.replace_pronouns(anon_input_seq, False)

        anon_input_seq = self.replace_numbers_and_months(anon_input_seq)

        return " ".join([x.strip() for x in anon_input_seq.split()])
    
    def ner(self, input_seq):
        orig_input_seq = deepcopy(input_seq)

        entities = self.get_identifiable_tokens(deepcopy(input_seq))
        
     

        entity2generic = self.get_entity_type_mapping(entities)
       

        anon_input_seq, url_map = self.replace_with_mapping(orig_input_seq, r"https?://\S+", "URL")
        
        anon_input_seq, user_map = self.replace_with_mapping(anon_input_seq, r"u\/[A-Za-z0-9_-]+", "USERNAME")
        
        anon_input_seq, email_map = self.replace_with_mapping(anon_input_seq, r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b", "EMAIL")

        anon_input_seq = self.replace_identified_entities( 
            entities, anon_input_seq, entity2generic
        )
        

        anon_input_seq, numeric_map = self.replace_numerics(anon_input_seq) 

        anon_input_seq = self.replace_pronouns(anon_input_seq, True) 

        anon_input_seq = self.replace_numbers_and_months(anon_input_seq, True) 

        anon_input_seq = self.inject_original_phrases(anon_input_seq, self.pronoun_map_reverse)

        anon_input_seq = self.inject_original_phrases(anon_input_seq, self.numeric_date)

        anon_input_seq = self.inject_original_phrases(anon_input_seq, url_map)

        anon_input_seq = self.inject_original_phrases(anon_input_seq, user_map)
        
        anon_input_seq = self.inject_original_phrases(anon_input_seq, email_map)

        anon_input_seq = self.inject_original_phrases(anon_input_seq, entity2generic)

        anon_input_seq = self.inject_original_phrases(anon_input_seq, numeric_map)

        return " ".join([x.strip() for x in anon_input_seq.split()])

    def supress(self, input_seq):
        
        anon_input_seq = self.ner(input_seq)

        pattern = r"\{'\s*[^']+\s*'\s*:\s*'\s*[^']+\s*'\}"

        result = re.sub(pattern, 'X', anon_input_seq)

        return result

    def generalize(self, input_seq):
        
        anon_input_seq = self.ner(input_seq)

        anon_input_seq = re.sub(self.getRegex('NUMERIC'), self.replacer(self.generate_interval), anon_input_seq)
        
        anon_input_seq = re.sub(self.getRegex('AGE'), self.replacer(self.generate_interval), anon_input_seq)
        
        anon_input_seq = re.sub(self.getRegex('NUMBER'), self.replacer(self.generate_interval), anon_input_seq)
        
        anon_input_seq = re.sub(self.getRegex('URL'), self.replacer(self.generalize_url), anon_input_seq)
        
        anon_input_seq = re.sub(self.getRegex('EMAIL'), self.replacer(self.gen_mail), anon_input_seq)
        
        anon_input_seq = re.sub(self.getRegex('PRONOUN'), self.replacer(self.generalize_pronoun), anon_input_seq)
        
        anon_input_seq = re.sub(self.getRegex('DATE'), self.replacer(self.generate_interval), anon_input_seq)
        
        anon_input_seq = re.sub(self.getRegex('ADDRESS'), self.replacer(self.generate_interval), anon_input_seq)
        
        anon_input_seq = re.sub(self.getRegex('[A-Z]+|[A-Z]+_[A-Z]+|MR/MS'), self.replacer(None), anon_input_seq)
        
        return anon_input_seq

        
    def randomize(self, input_seq):

        
        true_female_pronouns = random.randint(0, 1)
        
        true_male_pronouns = random.randint(0, 1)
        
        anon_input_seq = self.ner(input_seq)
        
        
        anon_input_seq = re.sub(self.getRegex('NUMERIC'), self.replacer(self.numeric_randomize, entity = 'NUMERIC'), anon_input_seq)
        
        anon_input_seq = re.sub(self.getRegex('AGE'), self.replacer(self.numeric_randomize, entity = 'AGE'), anon_input_seq)
        
        anon_input_seq = re.sub(self.getRegex('NUMBER'), self.replacer(self.numeric_randomize, entity = 'NUMBER'), anon_input_seq)
        
        anon_input_seq = re.sub(self.getRegex('PERSON_FIRSTNAME'), self.category_replacer("first_names"), anon_input_seq)
        
        anon_input_seq = re.sub(self.getRegex('PERSON_LASTNAME'), self.category_replacer("last_names"), anon_input_seq)
        
        anon_input_seq = re.sub(self.getRegex('LOCATION'), self.category_replacer("locations"), anon_input_seq)
        
        anon_input_seq = re.sub(self.getRegex('OCCUPATION'), self.category_replacer("occupations"), anon_input_seq)
        
        anon_input_seq = re.sub(self.getRegex('ORGANIZATION'), self.category_replacer("organizations"), anon_input_seq)
        
        anon_input_seq = re.sub(self.getRegex('PRONOUN'), self.replacer(self.pronoun_randomize, true_female_pronouns, true_male_pronouns), anon_input_seq)
        
        anon_input_seq = re.sub(self.getRegex('MR/MS'), self.replacer(self.mr_mrs_randomize, true_female_pronouns, true_male_pronouns), anon_input_seq)
        
        anon_input_seq = re.sub(self.getRegex('TITLE'), self.replacer(self.title_randomize, true_female_pronouns, true_male_pronouns), anon_input_seq)
        
        anon_input_seq = re.sub(self.getRegex('URL'), self.replacer(self.generate_random_url), anon_input_seq)
        
        anon_input_seq = re.sub(self.getRegex('EMAIL'), self.replacer(self.generate_random_email), anon_input_seq)
        
        anon_input_seq = re.sub(self.getRegex('USERNAME'), self.category_replacer("first_names", True), anon_input_seq)
        
        anon_input_seq = re.sub(self.getRegex('MONTH'), self.category_replacer("months"), anon_input_seq)
        
        anon_input_seq = re.sub(self.getRegex('DATE'), self.replacer(self.numeric_randomize, entity = 'DATE'), anon_input_seq)
        
        anon_input_seq = re.sub(self.getRegex('ADDRESS'), self.replacer(self.numeric_randomize, entity = 'ADDRESS'), anon_input_seq)
        
        anon_input_seq = re.sub(self.getRegex('[A-Z]+|[A-Z]+_[A-Z]+|MR/MS'), self.replacer(None), anon_input_seq)
        
        return anon_input_seq 
        





