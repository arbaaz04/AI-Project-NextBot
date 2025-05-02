import spacy
import os
os.environ['http_proxy'] = ''
os.environ['https_proxy'] = ''
os.environ['HTTP_PROXY'] = ''
os.environ['HTTPS_PROXY'] = ''
import json
import random
import nltk
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import yfinance as yf
import wikipedia
import requests
import datetime as dt
import re
from pathlib import Path
from spacy.tokens import DocBin

def contains_math_expression(text):
    cleaned = re.sub(r'[^\d\+\-\*/\(\)\.\s]', '', text)
    return bool(re.search(r'\d+\s*[\+\-\*/]\s*\(?\d+', cleaned))

import nltk
#nltk.download('wordnet')
import requests
requests.Session.proxies = {}
from sklearn.feature_extraction.text import TfidfVectorizer

class ChatbotModel(nn.Module):

    def __init__(self, input_size, output_size):
        super(ChatbotModel, self).__init__()

        self.fc1 = nn.Linear(input_size, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, output_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)

        return x


class ChatbotAssistant:

    def __init__(self, intents_path, function_mappings=None):
            self.model = None
            self.intents_path = intents_path
            self.documents = []
            self.vocabulary = []
            self.intents = []
            self.intents_responses = {}
            self.function_mappings = function_mappings
            self.X = None
            self.y = None
            self.vectorizer = TfidfVectorizer()
            self.last_topic_candidate = None
            self.user_name = None # Variable to store the user's name
            
            # Load the spaCy model for NER
            self.nlp = spacy.load("en_core_web_md")
            
            # Create custom entity extractor
            self.custom_entity_extractor = CustomEntityExtractor()
            
            # Create a more direct approach by just processing the doc after NER
            # This avoids issues with spaCy's component registration
            self.original_ner = self.nlp.get_pipe("ner")
            
            # Enhanced entity tracking
            self.entity_memory = {}
            self.context_history = []
            self.conversation_context = {
                'PERSON': [],
                'ORG': [],
                'GPE': [],
                'DATE': [],
                'TIME': [],
                'TOPIC': [],
                'PRODUCT': [],
                'EVENT': [],
                'PROGRAMMING_LANGUAGE': [],
                'FRAMEWORK': [],
                'DATABASE': [],
                'SUBJECT': []
            }
            
            # Conversation state tracking
            self.current_topic = None
            self.conversation_turns = 0
            self.recent_entities = {}  # Store most recently mentioned entities by type
            self.entity_relationships = {}  # Track relationships between entities
            
            # Coreference resolution
            self.coreference_resolver = CoreferenceResolver()

    @staticmethod
    def tokenize_and_lemmatize(text):
        lemmatizer = nltk.WordNetLemmatizer()

        words = nltk.word_tokenize(text)
        words = [lemmatizer.lemmatize(word.lower()) for word in words]

        return words

    #@staticmethod
    #def bag_of_words(self, words):
    #    return [1 if word in words else 0 for word in self.vocabulary]
    
    def parse_intents(self):
        lemmatizer = nltk.WordNetLemmatizer()

        if os.path.exists(self.intents_path):
            with open(self.intents_path, 'r') as file:
                intents_data = json.load(file)

            for intent in intents_data['intents']:
                if intent['tag'] not in self.intents:
                    self.intents.append(intent['tag'])
                    self.intents_responses[intent['tag']] = intent['responses']

                for pattern in intent['patterns']:
                    pattern_words = self.tokenize_and_lemmatize(pattern)
                    self.documents.append((' '.join(pattern_words), intent['tag']))  # Join words as a single string for TF-IDF

            # Now we will fit the vectorizer on the documents
            patterns = [doc[0] for doc in self.documents]  # Extract the text patterns
            self.vectorizer.fit(patterns)  # Fit the TF-IDF model
    
    def prepare_data(self):
        bags = []
        indices = []

        for doc in self.documents:
            # Use TF-IDF vectorizer to transform the document into a vector
            bag = self.vectorizer.transform([doc[0]]).toarray()  # Convert to numpy array
            intents_index = self.intents.index(doc[1])  # Get the index for the intent
            bags.append(bag)
            indices.append(intents_index)

        self.X = np.vstack(bags)
        self.y = np.array(indices)

    def train_model(self, batch_size, lr, epochs):
        print(f"Shape of input data (self.X): {self.X.shape}")
        X_tensor = torch.tensor(self.X.reshape(len(self.X), -1), dtype=torch.float32)
        y_tensor = torch.tensor(self.y, dtype=torch.long)
        dataset = TensorDataset(X_tensor, y_tensor)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        input_size = self.X.shape[1]  # Get the number of features in the TF-IDF representation
        self.model = ChatbotModel(input_size, len(self.intents))

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=lr)

        for epoch in range(epochs):
            running_loss = 0.0

            for batch_X, batch_y in loader:
                optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                running_loss += loss
            
            print(f"Epoch {epoch+1}: Loss: {running_loss / len(loader):.4f}")


    def save_model(self, model_path, dimensions_path):
        torch.save(self.model.state_dict(), model_path)

        with open(dimensions_path, 'w') as f:
            json.dump({ 'input_size': self.X.shape[1], 'output_size': len(self.intents) }, f)

    def load_model(self, model_path, dimensions_path):
        with open(dimensions_path, 'r') as f:
            dimensions = json.load(f)

            self.model = ChatbotModel(dimensions['input_size'], dimensions['output_size'])    
            self.model.load_state_dict(torch.load(model_path, weights_only=True))
            self.parse_intents()
            self.prepare_data()

            # Reinitialize the vectorizer with correct feature dimension
            input_size = self.X.shape[1]
            output_size = self.model.fc3.out_features
            self.model = ChatbotModel(input_size, output_size)
            self.model.load_state_dict(torch.load(model_path, weights_only=True))

    def process_message(self, input_message):
        # Resolve coreferences in the input message
        input_message = self.coreference_resolver.resolve(input_message, self.conversation_context)

        if contains_math_expression(input_message):
            print("Give me a moment to solve this.")
            return self.function_mappings['math'](input_message)
        
        words = self.tokenize_and_lemmatize(input_message)
        bag = self.vectorizer.transform([' '.join(words)]).toarray()  # Convert user input to a TF-IDF vector

        bag_tensor = torch.tensor(bag, dtype=torch.float32)

        self.model.eval()
        with torch.no_grad():
            predictions = self.model(bag_tensor)

        predicted_class_index = torch.argmax(predictions, dim=1).item()
        predicted_intent = self.intents[predicted_class_index]

        if predicted_intent == "wiki":
            # Clean the input for better Wikipedia search terms
            cleaned = input_message.lower().replace("who is", "").replace("tell me about", "").replace("what is", "").strip()
            self.last_topic_candidate = cleaned

        # Perform NER on the input message with spaCy's built-in NER
        doc = self.nlp(input_message)
        
        # Process custom entities separately (instead of as a spaCy component)
        doc = self.apply_custom_entity_extraction(doc)

        # Update coreference resolver with entities from the current message
        self.coreference_resolver.update_from_doc(doc)

        # Store user's name if identified and not already stored
        if doc.ents and self.user_name is None:
            for ent in doc.ents:
                if ent.label_ == "PERSON":
                    self.user_name = ent.text
                    break # Assume the first PERSON entity is the user's name

        if self.function_mappings and predicted_intent in self.function_mappings:
            # Return the result of the function mapping and the identified entities
            return self.function_mappings[predicted_intent](), doc.ents

        if self.intents_responses[predicted_intent]:
            # Return a random response from intents and the identified entities
            return random.choice(self.intents_responses[predicted_intent]), doc.ents
        else:
            # Return None response and the identified entities
            return None, doc.ents
            
    def apply_custom_entity_extraction(self, doc):
        """Apply custom entity extraction directly without using spaCy pipeline"""
        # Use our custom entity extractor
        return self.custom_entity_extractor(doc)

    def _update_entity_memory(self, entity):
        """Track entity frequency and context"""
        if entity.label_ not in self.entity_memory:
            self.entity_memory[entity.label_] = {}
            
        if entity.text not in self.entity_memory[entity.label_]:
            self.entity_memory[entity.label_][entity.text] = {
                'count': 0,
                'last_mentioned': None,
                'contexts': []
            }
            
        self.entity_memory[entity.label_][entity.text]['count'] += 1
        self.entity_memory[entity.label_][entity.text]['last_mentioned'] = dt.datetime.now()
        self.entity_memory[entity.label_][entity.text]['contexts'].append(str(entity.sent))

    def process_entities(self, doc):
        """Process and track entities"""
        entities = {
            'PERSON': [],
            'ORG': [],
            'GPE': [],
            'DATE': [],
            'PRODUCT': [],
            'EVENT': []
        }
        
        for ent in doc.ents:
            if ent.label_ in entities:
                entities[ent.label_].append(ent.text)
                self._update_entity_memory(ent)
        
        return entities

    def update_conversation_context(self, entities):
        """Update conversation context with newly identified entities"""
        for ent in entities:
            if ent.label_ in self.conversation_context:
                # Add entity if it's not already in the list
                if ent.text not in self.conversation_context[ent.label_]:
                    self.conversation_context[ent.label_].append(ent.text)
                
                # Update recent entities dictionary
                self.recent_entities[ent.label_] = ent.text
        
        # Increment conversation turn counter
        self.conversation_turns += 1
        
    def get_entity_based_response(self, intent, original_response, entities):
        """Generate a response that incorporates identified entities naturally"""
        modified_response = original_response
        
        # If we have no original response, don't try to modify it
        if not original_response or original_response == "__wiki__":
            return original_response
            
        # First handle any placeholder replacements
        if entities:
            for ent in entities:
                placeholder = f"[{ent.label_}]"
                if placeholder in modified_response:
                    modified_response = modified_response.replace(placeholder, ent.text, 1)
                elif "[ENTITY]" in modified_response:
                    modified_response = modified_response.replace("[ENTITY]", ent.text, 1)
        
        # If no placeholders were found but we have entities, consider enhancing the response
        # only if the original response doesn't already mention the entity
        if entities and all(ent.text not in modified_response for ent in entities):
            primary_entity = None
            
            # Prioritize PERSON entities for natural responses
            for ent in entities:
                if ent.label_ == "PERSON":
                    primary_entity = ent
                    break
                    
            if primary_entity:
                # Use different sentence structures based on intent
                if intent == "greeting":
                    prefixes = [
                        f"Hello {primary_entity.text}! ",
                        f"Hi there, {primary_entity.text}. ", 
                        f"Great to see you, {primary_entity.text}! "
                    ]
                    modified_response = random.choice(prefixes) + modified_response
                elif intent in ["wiki", "info"]:
                    modified_response = f"Let me tell you about {primary_entity.text}. " + modified_response
                else:
                    # For other intents, just acknowledge we recognized the entity
                    if random.random() < 0.5:  # Only do this sometimes to avoid being repetitive
                        modified_response = f"Regarding {primary_entity.text}, " + modified_response
        
        # Replace generic name with user's name if we have it
        if self.user_name:
            modified_response = modified_response.replace("Anas", self.user_name)
            
        return modified_response
    
    def extract_relations(self, doc):
        """Extract potential relationships between entities in the text"""
        relations = []
        
        # Get all entity pairs
        entities = list(doc.ents)
        for i, entity1 in enumerate(entities):
            for entity2 in entities[i+1:]:
                # Find the path of tokens between the entities
                if entity1.start < entity2.start:
                    start, end = entity1.end-1, entity2.start
                else:
                    start, end = entity2.end-1, entity1.start
                    
                # Extract the relationship text (simplified version)
                if start < end and end - start < 10:  # Limit to reasonable distances
                    relation_tokens = doc[start:end+1]
                    relation_text = relation_tokens.text
                    
                    # Store the relation
                    relations.append({
                        'entity1': (entity1.text, entity1.label_),
                        'entity2': (entity2.text, entity2.label_),
                        'relation': relation_text
                    })
                    
                    # Update entity relationships tracker
                    if entity1.text not in self.entity_relationships:
                        self.entity_relationships[entity1.text] = {}
                    if entity2.text not in self.entity_relationships[entity1.text]:
                        self.entity_relationships[entity1.text][entity2.text] = []
                    self.entity_relationships[entity1.text][entity2.text].append(relation_text)
                    
        return relations

class CustomEntityExtractor:
    """Custom entity extractor for domain-specific entities"""
    def __init__(self):
        # Domain-specific entity dictionaries
        self.custom_entities = {
            'PROGRAMMING_LANGUAGE': [
                'python', 'javascript', 'java', 'c++', 'c#', 'ruby', 'typescript',
                'php', 'swift', 'kotlin', 'go', 'rust', 'scala', 'dart'
            ],
            'FRAMEWORK': [
                'react', 'angular', 'vue', 'django', 'flask', 'spring', 'express',
                'laravel', 'rails', 'next.js', 'gatsby', 'tensorflow', 'pytorch'
            ],
            'DATABASE': [
                'sql', 'mongodb', 'mysql', 'postgresql', 'sqlite', 'oracle',
                'redis', 'cassandra', 'dynamodb', 'firebase', 'neo4j'
            ],
            'SUBJECT': [
                'math', 'science', 'history', 'geography', 'english', 'literature',
                'physics', 'chemistry', 'biology', 'economics', 'computer science'
            ]
        }
        
        # Compile regex patterns for each entity type
        self.patterns = {}
        for entity_type, terms in self.custom_entities.items():
            # Create case-insensitive word boundary patterns
            pattern = r'\b(' + '|'.join(map(re.escape, terms)) + r')\b'
            self.patterns[entity_type] = re.compile(pattern, re.IGNORECASE)
    
    def extract_entities(self, text):
        """Extract custom entities from text"""
        custom_entities = []
        
        # Extract entities for each type
        for entity_type, pattern in self.patterns.items():
            for match in pattern.finditer(text):
                start, end = match.span()
                entity_text = match.group()
                custom_entities.append({
                    'text': entity_text,
                    'label': entity_type,
                    'start': start,
                    'end': end
                })
                
        # Sort by start position
        custom_entities.sort(key=lambda x: x['start'])
        return custom_entities
        
    def __call__(self, doc):
        """Process a spaCy Doc and add custom entities"""
        custom_ents = self.extract_entities(doc.text)
        
        # Convert existing entities to a list we can modify
        ents = list(doc.ents)
        existing_spans = [(ent.start_char, ent.end_char) for ent in doc.ents]
        
        for ent in custom_ents:
            # Check if this entity overlaps with any existing entity
            overlap = any(
                max(ent['start'], start) < min(ent['end'], end)
                for start, end in existing_spans
            )
            
            if not overlap:
                # Find token span that corresponds to char span
                start_token = None
                end_token = None
                
                for i, token in enumerate(doc):
                    if token.idx <= ent['start'] < token.idx + len(token.text):
                        start_token = i
                    if token.idx <= ent['end'] <= token.idx + len(token.text):
                        end_token = i + 1
                        break
                
                if start_token is not None and end_token is not None:
                    # Create a new entity span
                    try:
                        from spacy.tokens import Span
                        new_ent = Span(doc, start_token, end_token, label=ent['label'])
                        ents.append(new_ent)
                        # Add to existing spans to avoid future overlaps
                        existing_spans.append((ent['start'], ent['end']))
                    except:
                        # Skip if there's an error creating the span
                        pass
        
        # Sort entities by start position
        ents = sorted(ents, key=lambda x: x.start)
        
        # Set the entities back on the doc
        doc.ents = ents
        return doc

# Create a proper spaCy component factory
def create_custom_entity_extractor(nlp, name):
    return CustomEntityExtractor()

class CoreferenceResolver:
    """Simple rule-based coreference resolver"""
    
    def __init__(self):
        self.last_entities = {}  # Store the most recent entity of each type
        self.pronoun_mappings = {
            'he': 'PERSON-M',
            'him': 'PERSON-M',
            'his': 'PERSON-M',
            'she': 'PERSON-F',
            'her': 'PERSON-F',
            'hers': 'PERSON-F',
            'it': 'THING',
            'its': 'THING',
            'they': 'GROUP',
            'them': 'GROUP',
            'their': 'GROUP',
            'these': 'NEAR-GROUP',
            'those': 'FAR-GROUP',
            'this': 'NEAR-THING',
            'that': 'FAR-THING'
        }
        self.entity_gender = {}  # Store inferred gender of entities
        
    def register_entity(self, entity_text, entity_type):
        """Register a new entity"""
        self.last_entities[entity_type] = entity_text
        
    def resolve(self, text, entities_dict):
        """Simple rule-based pronoun resolution"""
        # Skip if we don't have any entities in memory yet
        if not self.last_entities:
            return text
            
        # Find pronouns in text
        words = text.lower().split()
        for i, word in enumerate(words):
            if word in self.pronoun_mappings:
                entity_type = self.pronoun_mappings[word]
                
                # Try to find a matching entity
                resolved = None
                if entity_type == 'PERSON-M' and 'PERSON' in self.last_entities:
                    resolved = self.last_entities['PERSON']
                elif entity_type == 'PERSON-F' and 'PERSON' in self.last_entities:
                    resolved = self.last_entities['PERSON']
                elif entity_type == 'THING' and 'PRODUCT' in self.last_entities:
                    resolved = self.last_entities['PRODUCT']
                elif entity_type in ['GROUP', 'NEAR-GROUP', 'FAR-GROUP'] and 'ORG' in self.last_entities:
                    resolved = self.last_entities['ORG']
                elif entity_type in ['THING', 'NEAR-THING', 'FAR-THING'] and len(self.last_entities) > 0:
                    # Use the most recent non-person entity as fallback
                    for etype, etext in self.last_entities.items():
                        if etype != 'PERSON':
                            resolved = etext
                            break
                    if not resolved and len(self.last_entities) > 0:
                        # Use any entity as last resort
                        key = list(self.last_entities.keys())[0]
                        resolved = self.last_entities[key]
                
                # Replace the pronoun with the resolved entity
                if resolved:
                    words[i] = resolved
                    
        # Return the resolved text
        return ' '.join(words)
        
    def update_from_doc(self, doc):
        """Update entity tracking from a spaCy Doc"""
        for ent in doc.ents:
            self.register_entity(ent.text, ent.label_)
        
        return doc

class NERTrainer:
    def __init__(self, model=None):
        self.nlp = spacy.load(model) if model else spacy.blank("en")
        
        # Add NER pipe if it doesn't exist
        if "ner" not in self.nlp.pipe_names:
            ner = self.nlp.create_pipe("ner")
            self.nlp.add_pipe("ner", last=True)
        
    def prepare_training_data(self, training_data):
        """
        training_data format:
        [
            ("Apple is looking at buying U.K. startup for $1 billion", {
                "entities": [(0, 5, "ORG"), (27, 31, "GPE"), (44, 54, "MONEY")]
            }),
            ...
        ]
        """
        db = DocBin()
        for text, annotations in training_data:
            doc = self.nlp.make_doc(text)
            ents = []
            for start, end, label in annotations["entities"]:
                span = doc.char_span(start, end, label=label)
                if span:
                    ents.append(span)
            doc.ents = ents
            db.add(doc)
        return db

    def train(self, training_data, output_dir, n_iter=30):
        """Train the NER model"""
        Path(output_dir).mkdir(exist_ok=True)
        
        # Convert training data to spaCy format
        train_db = self.prepare_training_data(training_data)
        train_db.to_disk(Path(output_dir) / "train.spacy")
        
        # Configure training
        config = {
            "paths": {
                "train": str(Path(output_dir) / "train.spacy"),
                "dev": str(Path(output_dir) / "train.spacy")
            },
            "system": {"gpu_allocator": "pytorch"},
            "corpora": {"train": {"path": str(Path(output_dir) / "train.spacy")}}
        }
        
        # Train the model
        spacy.cli.train("config.cfg", output_dir, overrides=config)

def get_stocks():
    tickers = ['AAPL', 'META', 'NVDA', 'MSFT', 'GOOGL']
    stock_data = yf.download(tickers, period="1d", interval="1d")
    closing_prices = stock_data['Close'].iloc[-1]

    response = "📈 Here are the latest stock prices:\n"
    for ticker in tickers:
        price = closing_prices[ticker]
        response += f"{ticker}: ${price:.2f}\n"

    return response

def search_wikipedia():
    return "__wiki__"

def get_weather():
    city = input("🌤️ Enter your city: ")
    Base_URL = "http://api.openweathermap.org/data/2.5/weather?"
    api_key = '16e791bff0ec0642d76d8d4f1ca6ff0c'  # Replace with your actual OpenWeatherMap API key
    #url = f"http://api.openweathermap.org/data/2.5/weather?q={city}&appid={api_key}&units=metric"
    url = Base_URL + "appid=" + api_key + "&q=" + city
    response = requests.get(url).json()

    #print(response)  # Print the response for debugging
    def kelvin_to_celsius(kelvin):
        celcius, farenheit = kelvin - 273.15, (kelvin - 273.15) * 9/5 + 32
        return celcius, farenheit

    
    if response.get("cod") == 200:  # Check if the response code is OK (200)
        main = response["main"]
        weather = response["weather"][0]
        temperature = response["main"]["temp"]
        celsius, fahrenheit = kelvin_to_celsius(temperature)
        description = weather["description"]
        return f"🌤️ Weather in {city}: {description}, {celsius}°C"
    else:
        return "❌ City not found or invalid API request. Please try again."

def solve_math(expression=None):
    expression = input("🧮 Enter a math expression: ")
    
    # Extract only numbers/operators using regex
    match = re.findall(r'[\d\.\+\-\*/\(\)\s]+', expression)
    
    if not match:
        return "❌ Couldn't extract a valid math expression. Try again."

    clean_expr = ''.join(match)
    
    try:
        result = eval(clean_expr)
        return f"The result is: {result}"
    except:
        return "❌ Invalid math expression. Try again."
    
tomtom_api = 'ZZaSjIRF5HdKxjuEAQEKAqyfu7Zgxzpx'
    
def geocode_address(address, api_key):
    url = f"https://api.tomtom.com/search/2/geocode/{address}.json"
    params = {"key": api_key}
    res = requests.get(url, params=params)
    data = res.json()
    position = data['results'][0]['position']
    return position['lat'], position['lon']

def get_traffic_info(lat, lon, api_key):
    url = "https://api.tomtom.com/traffic/services/4/flowSegmentData/relative0/10/json"
    params = {
        "point": f"{lat},{lon}",
        "key": api_key
    }
    res = requests.get(url, params=params)
    if res.status_code == 200:
        return res.json()['flowSegmentData']
    else:
        return None
    
def get_location():
    res = requests.get("https://ipwho.is/")
    data = res.json()
    if data['success']:
        lat = data['latitude']
        lon = data['longitude']
        city = data['city']
        return lat, lon, city
    else:
        raise Exception("Location fetch failed")

def get_traffic():
    address = input('Where would you want the traffic information of?: ')
    lat, lon = geocode_address(address, tomtom_api)
    traffic = get_traffic_info(lat, lon, tomtom_api)
    if traffic:
        current_speed = traffic['currentSpeed']
        free_flow_speed = traffic['freeFlowSpeed']
        congestion_level = ""

        if current_speed >= 0.9 * free_flow_speed:
            congestion_level = "Traffic is flowing smoothly."
        elif current_speed >= 0.6 * free_flow_speed:
            congestion_level = "There is moderate traffic."
        else:
            congestion_level = "Heavy traffic or congestion detected."

        response = congestion_level + f" Current speed is {current_speed}."
    else:
        response = "Could not get traffic info."
    return response

if __name__ == '__main__':
    wiki_mode = False
    last_topic = None

    assistant = ChatbotAssistant('intents.json', function_mappings={'stocks': get_stocks, 'wiki': search_wikipedia, 'weather': get_weather, 'math': solve_math, 'traffic': get_traffic})

    #assistant.load_model('chatbot_model.pth', 'dimensions.json')
    assistant.parse_intents()
    assistant.prepare_data()
    assistant.train_model(batch_size=8, lr=0.001, epochs=100)

    print("Chatbot initialized and ready! Type /quit to exit.")
    
    while True:
        message = input('Enter your message: ')

        if message.lower() == '/quit':
            print("Goodbye! Have a great day!")
            break
                
        # Check for weather-related queries
        if message.lower() in ["what's the weather like?", "weather update"]:
            print(get_weather())  # Call the weather function directly
            continue
            
        # Process message with NER
        doc = assistant.nlp(message)
        
        # Extract relations between entities
        relations = assistant.extract_relations(doc)
        
        # Wiki mode handling
        if wiki_mode:
            follow_up_keywords = ['his', 'her', 'their', 'age', 'companies', 'net worth', 'more', 'about']
            if any(word in message.lower() for word in follow_up_keywords):
                topic = assistant.last_topic_candidate or last_topic
                topic += " " + message.lower()  # Explicitly add the follow-up topic (e.g., "Elon Musk age")
            else:
                topic = message
            print("🔍 Searching Wikipedia...")
            try:
                summary = wikipedia.summary(topic, sentences=2)
                print(f"📚 {summary}")
                last_topic = topic
                
                # Update entity memory with information from Wikipedia response
                wiki_doc = assistant.nlp(summary)
                for ent in wiki_doc.ents:
                    assistant._update_entity_memory(ent)
                    
            except wikipedia.exceptions.DisambiguationError as e:
                print(f"❗ Too many results. Try being more specific. Options: {e.options[:5]}")
            except wikipedia.exceptions.PageError:
                print("❌ I couldn't find any information on that. Try another question.")
            wiki_mode = False
            continue

        # Regular message processing
        response, entities = assistant.process_message(message)
        
        # Update conversation context with the new entities
        assistant.update_conversation_context(entities)
        
        # Choose intent-aware response based on entities
        predicted_class_index = torch.argmax(
            assistant.model(torch.tensor(assistant.vectorizer.transform([' '.join(assistant.tokenize_and_lemmatize(message))]).toarray(), 
                                         dtype=torch.float32)), 
            dim=1).item()
        intent = assistant.intents[predicted_class_index]
        
        # Get enhanced response with natural entity handling
        enhanced_response = assistant.get_entity_based_response(intent, response, entities)

        # Handle wiki mode switch and final response formatting
        if enhanced_response == "__wiki__":
            wiki_mode = True
            print("🧠 Sure! What would you like to learn about?")
        elif enhanced_response:
            # Add a reference to past entities occasionally to make conversation more connected
            if assistant.conversation_turns > 1 and random.random() < 0.3 and assistant.recent_entities:
                # Choose a random entity type we've seen before
                entity_types = list(assistant.recent_entities.keys())
                if entity_types:
                    chosen_type = random.choice(entity_types)
                    entity_text = assistant.recent_entities[chosen_type]
                    
                    # Only add if entity isn't already in the response
                    if entity_text not in enhanced_response:
                        recall_phrases = [
                            f"Going back to {entity_text} we talked about earlier, ",
                            f"By the way, regarding {entity_text}, ",
                            f"Speaking of {entity_text}, "
                        ]
                        enhanced_response = random.choice(recall_phrases) + enhanced_response
            
            print(enhanced_response)
        else:
            print("I'm not sure how to respond to that.")
            
        # Debug info (optional)
        # If you want to see identified entities and relations, uncomment:
        # if entities:
        #    print("\nIdentified entities:")
        #    for ent in entities:
        #        print(f"- {ent.text}: {ent.label_}")
        # if relations:
        #    print("\nIdentified relations:")
        #    for rel in relations:
        #        print(f"- {rel['entity1'][0]} {rel['relation']} {rel['entity2'][0]}")