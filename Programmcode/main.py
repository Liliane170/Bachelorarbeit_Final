from transformers import AutoTokenizer, AutoModelForTokenClassification
import torch
from transformers import pipeline
import json
from cryptography.fernet import Fernet
from transformers.models.deprecated import qdqbert
from anonymizer import Anonymizer
from readability import Readability
from analyse import Analyser
from informationContent import InformationContent
from collectData import Collecter


if __name__ == "__main__":


    def encrypt (data, secret_key_path, data_path):
    
        key = Fernet.generate_key()

        with open(secret_key_path, "wb") as key_file: 
            key_file.write(key)
     
        fernet = Fernet(key)
    
        encrypted_data = fernet.encrypt(data)

        with open(data_path, "wb") as encrypted_file:
            encrypted_file.write(encrypted_data)
    
    def decrypt(secret_key_path, data_path):
    
        with open(secret_key_path, "rb") as keyfile: 
            key = keyfile.read()

        fernet = Fernet(key)
    
        # Verschlüsselte Daten laden
        with open(data_path, "rb") as f:
            encrypted_data = f.read()
    
        # Entschlüsseln
        decrypted_data = fernet.decrypt(encrypted_data)
    
        # Bytes -> String -> JSON laden
        return json.loads(decrypted_data.decode('utf-8'))
    
    def collectPosts():
        
        collecter = Collecter()
        
        posts = collecter.getPosts()
        
        #Speicherort für die verschlüsselte json, welche die gesammelten Posts enthält, und für den Key eintragen!!! 
        encrypt(posts, "secret_raw_posts.key", "raw_posts.enc") 
    
        def anonymize():
    
        # Pfad zum NER Modell 
        # runterladen unter https://drive.google.com/file/d/1YBccngYE3lvod87TI6UIhBzrN7nY9vHS/view?usp=sharing
        # der Pfad sollte direkt in ./data/en zeigen und nicht auf das übergeordnete Verzeichnis
        tokenizer = AutoTokenizer.from_pretrained("./data/en")    
        model = AutoModelForTokenClassification.from_pretrained("./data/en") 
        
        classifier = pipeline("ner", model=model, tokenizer=tokenizer, device=torch.device("cpu"))
        
        #Speicherort für die verschlüsselte json, welche die gesammelten Posts enthält, und für den Key eintragen!!! 
        posts = decrypt("secret_raw_posts.key", "raw_posts.enc") 
        

        anonymizer = Anonymizer(classifier)

   
        outputs = []
        for idx, text in enumerate(posts):
            generalized = anonymizer.generalize(text)
            tagged = anonymizer.tag(text)
            suppressed = anonymizer.supress(text)
            randomized = anonymizer.randomize(text)
                
            output = {
                "original": text,
                "generalized": generalized,
                "tagged": tagged,
                "suppressed": suppressed,
                "randomized": randomized
            }

            outputs.append(output)
     
            print(f"Anonymized {idx + 1}/{len(posts)}")
            
            
        json_string = json.dumps(outputs, ensure_ascii=False, indent=2)
        
        #Speicherort für die verschlüsselte json, welche die anonymisierten Posts enthält, und für den Key eintragen!!! 
        encrypt(json_string.encode('utf-8'), "secret_anonymized_posts.key", "anonymized_posts.enc") 
        
        print("Anonymized")        

    def measure_readability():
        
        readability = Readability()
        
        #Speicherort für die verschlüsselte json, welche die anonymisierten Posts enthält, und für den Key eintragen!!! 
        posts = decrypt("secret_anonymized_posts.key", "anonymized_posts.enc") 
        
        outputs = []
        
        for idx, post in enumerate(posts):
            
            flesch_original = readability.flesch(post["original"])
            gunning_original = readability.gunning(post["original"])
            dale_chall_original = readability.dale_chall(post["original"])
            flesch_generalized = readability.flesch(post["generalized"])
            gunning_generalized = readability.gunning(post["generalized"])
            dale_chall_generalized = readability.dale_chall(post["generalized"])
            flesch_tagged = readability.flesch(post["tagged"])
            gunning_tagged = readability.gunning(post["tagged"])
            dale_chall_tagged = readability.dale_chall(post["tagged"])
            flesch_suppressed = readability.flesch(post["suppressed"])
            gunning_suppressed = readability.gunning(post["suppressed"])
            dale_chall_suppressed = readability.dale_chall(post["suppressed"])
            flesch_randomized = readability.flesch(post["randomized"])
            gunning_randomized = readability.gunning(post["randomized"])
            dale_chall_randomized = readability.dale_chall(post["randomized"])

            output = {
                "original": post["original"],
                "generalized": post["generalized"],
                "tagged": post["tagged"],
                "suppressed": post["suppressed"],
                "randomized": post["randomized"],
                "flesch_original": flesch_original,
                "gunning_original": gunning_original,
                "dale_chall_original": dale_chall_original,
                "flesch_generalized": flesch_generalized,
                "gunning_generalized": gunning_generalized,
                "dale_chall_generalized": dale_chall_generalized,
                "flesch_tagged": flesch_tagged,
                "gunning_tagged": gunning_tagged,
                "dale_chall_tagged": dale_chall_tagged,
                "flesch_suppressed": flesch_suppressed,
                "gunning_suppressed": gunning_suppressed,
                "dale_chall_suppressed": dale_chall_suppressed,
                "flesch_randomized": flesch_randomized,
                "gunning_randomized": gunning_randomized,
                "dale_chall_randomized": dale_chall_randomized
            }

            outputs.append(output)

        json_string = json.dumps(outputs, ensure_ascii=False, indent=2)
            
        #Speicherort für die verschlüsselte json, welche die Posts und deren Lesbarkeit enthält, und für den Key eintragen!!! 
        encrypt(json_string.encode('utf-8'), "secret_posts_readability.key", "posts_readability.enc") 

        print("Readability measured") 

    def measure_information_content():

        informationContent = InformationContent()

        #Speicherort für die verschlüsselte json, welche die Posts und deren Lesbarkeit enthält, und für den Key eintragen!!! 
        posts = decrypt("secret_posts_readability.key", "posts_readability.enc") 

        original = informationContent.measure_information_content_for_category(posts, "original")

        generalized = informationContent.measure_information_content_for_category(posts, "generalized")

        randomized = informationContent.measure_information_content_for_category(posts, "randomized")

        tagged = informationContent.measure_information_content_for_category(posts, "tagged")

        suppressed = informationContent.measure_information_content_for_category(posts, "suppressed")

        results = {
            "original": original,
            "generalized": generalized,
            "randomized": randomized,
            "tagged": tagged,
            "suppressed": suppressed
        }


        with open("information_content_results.json", "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)

        print("Entropy measured") 
            
      



