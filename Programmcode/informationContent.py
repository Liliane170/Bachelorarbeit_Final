import re
import math


class InformationContent():
 
    
    def measure_information_content_for_category(self, data, category):

        word_counts = {}
        total_words = 0

        for entry in data:
            words = re.findall(r'\b(?:https?://\S+|[a-zA-Z0-9_/]+)\b', entry[category].lower())
            for word in words:
                if word in word_counts:
                    word_counts[word] += 1
                else:
                    word_counts[word] = 1
                total_words += 1

        entropy = 0

        for word_count in word_counts:
            entropy = entropy + ((word_counts[word_count]/total_words) * math.log2((word_counts[word_count]/total_words)))

        entropy = -1 * entropy
                
        return entropy 

 