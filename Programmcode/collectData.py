import requests
import re
import html
import emoji
import json

class Collecter():
 
    
    def getPosts(self):

        #Notwendige Schritte für Zugriff auf API 
            #1. Reddit Konto erstellen unter https://www.reddit.com/
            #2. Application erstellen unter https://www.reddit.com/prefs/apps/

        #Komplettes Tutorial unter https://www.youtube.com/watch?v=FdjVoOf9HN4

        #Hier Credentials der API eintragen
        CLIENT_ID = 'Client_id'
        SECRET_KEY = 'Secret_key'

        auth = requests.auth.HTTPBasicAuth(CLIENT_ID, SECRET_KEY)

        #Hier Anmeldedaten des Reddit Kontos eintragen
        data = {
            'grant_type': 'password',
            'username': 'username',
            'password': r"password"
        }

        headers = {'User-Agent': 'MyAPI'}

        res = requests.post(
            'https://www.reddit.com/api/v1/access_token',
            auth = auth,
            data = data,
            headers = headers
        )

        TOKEN = res.json()['access_token']

        headers['Authorization'] = f'bearer {TOKEN}'

        params = {
            "t": "all",     
            "limit": 100   
        }

        res_relationships = requests.get('https://oauth.reddit.com/r/relationships/top.json',
                         headers = headers,
                         params = params                 
        )

        res_legal = requests.get('https://oauth.reddit.com/r/legaladvice/top.json',
                         headers = headers,
                         params = params                 
        )

        data_relationships = res_relationships.json()

        data_legal = res_legal.json()
    
        posts = []

        for post in data_relationships["data"]["children"]:
            post_data = post["data"]
            text = post_data["selftext"]

            if text:  # Nur echte Text-Posts
                
                posts.append(self.clean_text(text))

        for post in data_legal["data"]["children"]:
            post_data = post["data"]
            text = post_data["selftext"]

            if text:  # Nur echte Text-Posts
                
                posts.append(self.clean_text(text))
            
    
        json_string = json.dumps(posts, ensure_ascii=False, indent=2)
    
        return json_string.encode('utf-8')
        

    def clean_text(self, text):   
    
    
        # &amp;#x200B; wegmachen
        text = html.unescape(text)
        text = html.unescape(text) 
    
        # Zeilenumbrüche ersetzen
        text = text.replace("\n", " ")
    

        #Markdown Tags entfernen

        # Markdown-Links entfernen: [Text](URL) ? Text
        text = re.sub(r'\[([^\]]+)\]\([^)]+\)', r'\1', text)
    
        # Entferne den Backslash vor "
        text = text.replace(r'\\"', '"')

        # Inline-Code entfernen: `code`
        text = re.sub(r'`[^`]+`', '', text)

        # Fettdruck/Kursiv entfernen: **text**, *text*, __text__, _text_
        text = re.sub(r'(\*\*|__|\*|_)(.*?)\1', r'\2', text)

        # Überschriften entfernen: # Text
        text = re.sub(r'#+\s*', '', text)

        # Zitate entfernen: > Text
        text = re.sub(r'>\s*', '', text)

        # Listenpunkte entfernen
        text = re.sub(r'^[-*+]\s+', '', text, flags=re.MULTILINE)

    

        # Mehrfache Leerzeichen zu einem Leerzeichen
        text = re.sub(r'\s+', ' ', text)

        #Emojis entfernen
    
        text = emoji.replace_emoji(text, replace='')    

        return text.strip()
